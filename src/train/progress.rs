use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::path::Path;

#[derive(Clone, Copy)]
struct Row {
    epoch: f64,
    box_loss: f64,
    cls_loss: f64,
    dfl_loss: f64,
    recall: Option<f64>,
    precision: Option<f64>,
    map50: Option<f64>,
    map: Option<f64>,
}

fn column(header: &[&str], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|value| *value == name)
        .with_context(|| format!("training history is missing the `{name}` column"))
}

fn number(fields: &[&str], index: usize, name: &str) -> Result<f64> {
    fields
        .get(index)
        .with_context(|| format!("training history row is missing `{name}`"))?
        .parse::<f64>()
        .with_context(|| format!("training history contains an invalid `{name}` value"))
}

fn read_history(path: &Path) -> Result<Vec<Row>> {
    let csv = std::fs::read_to_string(path)
        .with_context(|| format!("reading training history {}", path.display()))?;
    let mut lines = csv.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .context("training history contains no header")?
        .split(',')
        .collect::<Vec<_>>();
    let epoch = column(&header, "epoch")?;
    let box_loss = column(&header, "box")?;
    let cls_loss = column(&header, "cls")?;
    let dfl_loss = column(&header, "dfl")?;
    let recall = column(&header, "Recall")?;
    let precision = column(&header, "Precision")?;
    let map50 = column(&header, "mAP@50")?;
    let map = column(&header, "mAP")?;
    let evaluated = header.iter().position(|value| *value == "evaluated");

    lines
        .map(|line| {
            let fields = line.split(',').collect::<Vec<_>>();
            let has_validation = evaluated
                .and_then(|index| fields.get(index))
                .is_none_or(|value| *value == "true");
            let metric = |index, name| {
                has_validation
                    .then(|| number(&fields, index, name))
                    .transpose()
            };
            Ok(Row {
                epoch: number(&fields, epoch, "epoch")?,
                box_loss: number(&fields, box_loss, "box")?,
                cls_loss: number(&fields, cls_loss, "cls")?,
                dfl_loss: number(&fields, dfl_loss, "dfl")?,
                recall: metric(recall, "Recall")?,
                precision: metric(precision, "Precision")?,
                map50: metric(map50, "mAP@50")?,
                map: metric(map, "mAP")?,
            })
        })
        .collect()
}

struct Panel<'a> {
    title: &'a str,
    top: f64,
    bottom: f64,
    series: &'a [(&'a str, &'a str, fn(&Row) -> Option<f64>)],
}

fn draw_panel(svg: &mut String, rows: &[Row], panel: Panel<'_>, x0: f64, width: f64) {
    let height = panel.bottom - panel.top;
    let min_epoch = rows.first().map_or(0.0, |row| row.epoch);
    let max_epoch = rows.last().map_or(1.0, |row| row.epoch);
    let epoch_span = (max_epoch - min_epoch).max(1.0);
    let observed_max = panel
        .series
        .iter()
        .flat_map(|(_, _, value)| rows.iter().filter_map(*value))
        .fold(0.0_f64, f64::max);
    let y_max = if panel.title == "Validation metrics" {
        (observed_max * 1.1).clamp(0.1, 1.0)
    } else {
        (observed_max * 1.08).max(1e-6)
    };
    let x = |epoch: f64| x0 + (epoch - min_epoch) / epoch_span * width;
    let y = |value: f64| panel.bottom - value / y_max * height;

    let _ = writeln!(
        svg,
        r#"<text x="{x0}" y="{}" class="title">{}</text>"#,
        panel.top - 24.0,
        panel.title
    );
    for tick in 0..=5 {
        let fraction = tick as f64 / 5.0;
        let py = panel.bottom - fraction * height;
        let value = fraction * y_max;
        let _ = writeln!(
            svg,
            r#"<line x1="{x0}" y1="{py:.1}" x2="{}" y2="{py:.1}" class="grid"/><text x="{}" y="{:.1}" class="tick" text-anchor="end">{value:.3}</text>"#,
            x0 + width,
            x0 - 10.0,
            py + 4.0
        );
    }
    let _ = writeln!(
        svg,
        r#"<line x1="{x0}" y1="{}" x2="{x0}" y2="{}" class="axis"/><line x1="{x0}" y1="{}" x2="{}" y2="{}" class="axis"/>"#,
        panel.top,
        panel.bottom,
        panel.bottom,
        x0 + width,
        panel.bottom
    );
    for tick in 0..=5 {
        let fraction = tick as f64 / 5.0;
        let epoch = min_epoch + fraction * epoch_span;
        let px = x0 + fraction * width;
        let _ = writeln!(
            svg,
            r#"<text x="{px:.1}" y="{}" class="tick" text-anchor="middle">{epoch:.0}</text>"#,
            panel.bottom + 22.0
        );
    }

    for (series_index, (name, color, value)) in panel.series.iter().enumerate() {
        let points = rows
            .iter()
            .filter_map(|row| {
                value(row).map(|value| format!("{:.1},{:.1}", x(row.epoch), y(value)))
            })
            .collect::<Vec<_>>()
            .join(" ");
        if !points.is_empty() {
            let _ = writeln!(
                svg,
                r#"<polyline points="{points}" fill="none" stroke="{color}" class="curve"/>"#
            );
        }
        let legend_x = x0 + width - 155.0 * (panel.series.len() - series_index) as f64;
        let legend_y = panel.top - 24.0;
        let _ = writeln!(
            svg,
            r#"<line x1="{legend_x:.1}" y1="{legend_y:.1}" x2="{:.1}" y2="{legend_y:.1}" stroke="{color}" class="curve"/><text x="{:.1}" y="{:.1}" class="legend">{name}</text>"#,
            legend_x + 24.0,
            legend_x + 30.0,
            legend_y + 4.0
        );
    }
}

/// Render the numeric training history as a standalone SVG without a plotting dependency.
pub fn write_training_progress_svg(history: &Path, output: &Path) -> Result<()> {
    let rows = read_history(history)?;
    anyhow::ensure!(!rows.is_empty(), "training history contains no epochs");
    let losses: &[(&str, &str, fn(&Row) -> Option<f64>)] = &[
        ("box", "#2563eb", |row| Some(row.box_loss)),
        ("class", "#dc2626", |row| Some(row.cls_loss)),
        ("DFL", "#16a34a", |row| Some(row.dfl_loss)),
    ];
    let metrics: &[(&str, &str, fn(&Row) -> Option<f64>)] = &[
        ("mAP", "#7c3aed", |row| row.map),
        ("mAP@50", "#ea580c", |row| row.map50),
        ("precision", "#0891b2", |row| row.precision),
        ("recall", "#4d7c0f", |row| row.recall),
    ];
    let mut svg = String::from(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">
<style>.background{fill:#fff}.axis{stroke:#111;stroke-width:1.5}.grid{stroke:#ddd;stroke-width:1}.curve{stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}.title{font-family:sans-serif;font-size:20px;font-weight:600;fill:#111}.tick{font-family:sans-serif;font-size:13px;fill:#444}.legend{font-family:sans-serif;font-size:14px;fill:#222}.label{font-family:sans-serif;font-size:15px;fill:#222}</style>
<rect class="background" width="1200" height="820"/>
<text x="600" y="34" text-anchor="middle" class="title">YOLO training progress</text>
"#,
    );
    draw_panel(
        &mut svg,
        &rows,
        Panel {
            title: "Training losses",
            top: 85.0,
            bottom: 345.0,
            series: losses,
        },
        85.0,
        1065.0,
    );
    draw_panel(
        &mut svg,
        &rows,
        Panel {
            title: "Validation metrics",
            top: 470.0,
            bottom: 730.0,
            series: metrics,
        },
        85.0,
        1065.0,
    );
    svg.push_str(
        r#"<text x="600" y="790" text-anchor="middle" class="label">Global epoch</text>
</svg>
"#,
    );
    let temporary = output.with_extension("svg.tmp");
    std::fs::write(&temporary, svg)?;
    std::fs::rename(temporary, output)?;
    Ok(())
}
