"""Generate a clean vector architecture diagram for Figure 1."""

from math import atan2, cos, sin
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "figures" / "Fig1_CA_DTNet_architecture.pdf"

PAGE_W, PAGE_H = 1200, 650
NAVY = "#183B56"
BLUE = "#2E6F9E"
PALE = "#EEF5FA"
PALE_ALT = "#F7F9FB"
TEXT = "#17212B"
ARROW = "#365B73"


def color(hex_value):
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def draw_centered_lines(c, lines, x, y, width, font="Helvetica", size=10,
                        leading=13, fill=TEXT):
    c.setFillColorRGB(*color(fill))
    c.setFont(font, size)
    current_y = y
    for line in lines:
        while stringWidth(line, font, size) > width and " " in line:
            words = line.split()
            split_at = len(words) - 1
            while split_at > 1 and stringWidth(" ".join(words[:split_at]), font, size) > width:
                split_at -= 1
            first = " ".join(words[:split_at])
            rest = " ".join(words[split_at:])
            c.drawCentredString(x, current_y, first)
            current_y -= leading
            line = rest
        c.drawCentredString(x, current_y, line)
        current_y -= leading


def panel(c, x, title, number, fill):
    c.setFillColorRGB(*color(fill))
    c.setStrokeColorRGB(*color("#B7C8D4"))
    c.setLineWidth(1.2)
    c.roundRect(x, 28, 276, 586, 12, fill=1, stroke=1)
    c.setFillColorRGB(*color(NAVY))
    c.roundRect(x + 10, 552, 256, 48, 8, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 19)
    c.drawCentredString(x + 31, 568, number)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(x + 145, 568, title)


def box(c, x, y, w, h, title, lines, fill="#FFFFFF", title_size=17.5, body_size=14.5):
    c.setFillColorRGB(*color(fill))
    c.setStrokeColorRGB(*color(BLUE))
    c.setLineWidth(1.4)
    c.roundRect(x, y, w, h, 9, fill=1, stroke=1)
    c.setFillColorRGB(*color(NAVY))
    c.setFont("Helvetica-Bold", title_size)
    c.drawCentredString(x + w / 2, y + h - 25, title)
    draw_centered_lines(c, lines, x + w / 2, y + h - 48, w - 18,
                        size=body_size, leading=17)
    return x, y, w, h


def left(rect, fraction=0.5):
    x, y, _w, h = rect
    return x, y + h * fraction


def right(rect, fraction=0.5):
    x, y, w, h = rect
    return x + w, y + h * fraction


def top(rect, fraction=0.5):
    x, y, w, h = rect
    return x + w * fraction, y + h


def bottom(rect, fraction=0.5):
    x, y, w, _h = rect
    return x + w * fraction, y


def arrow(c, points):
    c.setStrokeColorRGB(*color(ARROW))
    c.setFillColorRGB(*color(ARROW))
    c.setLineWidth(1.55)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, fill=0, stroke=1)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    angle = atan2(y1 - y0, x1 - x0)
    length = 9
    spread = 0.48
    p2 = (x1 - length * cos(angle - spread), y1 - length * sin(angle - spread))
    p3 = (x1 - length * cos(angle + spread), y1 - length * sin(angle + spread))
    head = c.beginPath()
    head.moveTo(x1, y1)
    head.lineTo(*p2)
    head.lineTo(*p3)
    head.close()
    c.drawPath(head, fill=1, stroke=0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("CA-DTNet architecture")

    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColorRGB(*color(NAVY))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(PAGE_W / 2, 626,
                        "Trace-driven communication-aware traffic digital twin")

    xs = [18, 312, 606, 900]
    panel(c, xs[0], "Measured inputs", "1", PALE_ALT)
    panel(c, xs[1], "Trace replay", "2", PALE)
    panel(c, xs[2], "CA-DTNet", "3", PALE_ALT)
    panel(c, xs[3], "Model outputs", "4", PALE)

    traffic = box(c, 43, 430, 226, 92, "pNEUMA traffic",
                  ["1-s trajectories", "cell-level traffic states"])
    comm = box(c, 43, 272, 226, 92, "Measured V2X traces",
               ["CICV5G and SEE-V2X", "latency, loss, jitter, radio/load"])
    split = box(c, 43, 112, 226, 92, "Evaluation protocol",
                ["chronological 70/15/15 split", "seeds 42, 52, and 62"])

    graph = box(c, 337, 430, 226, 92, "Traffic graph",
                ["40-m cells and adjacency", "60-step input windows"])
    replay = box(c, 337, 272, 226, 92, "Contiguous trace replay",
                 ["seeded trace assignment", "ordered measured segments"])
    impaired = box(c, 337, 112, 226, 92, "Impaired observation",
                   ["loss -> sample-and-hold", "latency -> AoI; metrics -> context"])

    encoders = box(c, 631, 420, 226, 114, "Dual input encoders",
                   ["traffic-state projection", "communication-context embedding"],
                   fill="#FFFFFF")
    fusion = box(c, 631, 292, 226, 92, "Reliability gate and fusion",
                 ["communication-conditioned", "traffic representation"], fill="#EAF3F9")
    core = box(c, 631, 164, 226, 92, "Spatiotemporal core",
               ["GRU + masked graph attention", "multi-horizon decoder"])
    objective = box(c, 631, 62, 226, 70, "Joint supervision",
                    ["quantile loss + 0.3 x reconstruction MAE"], fill="#FFF8E8",
                    title_size=16.5, body_size=12.5)

    recon = box(c, 925, 430, 226, 92, "State reconstruction",
                ["clean final traffic state", "reconstruction metrics"])
    forecast = box(c, 925, 272, 226, 92, "Probabilistic forecast",
                   ["0.1 / 0.5 / 0.9 quantiles", "5-s, 10-s, and 30-s horizons"])
    evaluation = box(c, 925, 112, 226, 92, "Evaluation",
                     ["accuracy and coverage", "domain sensitivity and efficiency"])

    arrow(c, [right(traffic), left(graph)])
    arrow(c, [right(comm), left(replay)])
    arrow(c, [bottom(replay), top(impaired)])
    arrow(c, [right(graph), (596, 476), (596, 496), left(encoders, 0.68)])
    arrow(c, [right(replay), (585, 318), (585, 455), left(encoders, 0.32)])
    arrow(c, [bottom(encoders), top(fusion)])
    arrow(c, [bottom(fusion), top(core)])
    arrow(c, [bottom(core), top(objective)])
    arrow(c, [right(core, 0.72), (880, 230), (880, 476), left(recon)])
    arrow(c, [right(core, 0.50), (894, 210), (894, 318), left(forecast)])
    arrow(c, [bottom(recon), (1038, 396), (1180, 396), (1180, 158), right(evaluation)])
    arrow(c, [bottom(forecast), top(evaluation)])
    arrow(c, [right(split), (295, 158), (295, 76), (618, 76), left(objective, 0.20)])

    c.setFillColorRGB(*color("#516672"))
    c.setFont("Helvetica", 8.7)
    c.drawCentredString(744, 40,
                        "Forecasting and reconstruction share the learned latent representation.")

    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
