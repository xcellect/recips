const COLORS = {
  Recon: "#86a8ff",
  Ipsundrum: "#f0a245",
  "Ipsundrum+affect": "#6bc0a8",
  default: "#d8dde4",
};

function modelColor(model) {
  return COLORS[model] || COLORS.default;
}

function select(id) {
  return document.getElementById(id);
}

function metricLabel(labels, key) {
  return labels[key] || key.replaceAll("_", " ");
}

function populateSelect(node, options, selected) {
  node.innerHTML = "";
  options.forEach((option) => {
    const el = document.createElement("option");
    el.value = option.value;
    el.textContent = option.label;
    if (option.value === selected) {
      el.selected = true;
    }
    node.appendChild(el);
  });
}

function setLink(id, href) {
  const node = select(id);
  if (node && href) {
    node.href = href;
  }
}

function setImage(id, src) {
  const node = select(id);
  if (node && src) {
    node.src = src;
  }
}

function renderHeadlineMetrics(metrics) {
  const root = select("headline-metrics");
  root.innerHTML = "";
  metrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="metric-value">${metric.value}${metric.suffix || ""}</div>
      <div class="metric-label">${metric.label}</div>
      <div class="metric-context">${metric.context}</div>
    `;
    root.appendChild(card);
  });
}

function renderGallery(items) {
  const root = select("gallery-grid");
  root.innerHTML = "";
  items.forEach((item) => {
    const article = document.createElement("article");
    article.className = "panel gallery-card";
    article.innerHTML = `
      <img src="${item.src}" alt="${item.title}">
      <h3>${item.title}</h3>
      <p>${item.caption}</p>
    `;
    root.appendChild(article);
  });
}

function renderConfig(config) {
  const root = select("config-list");
  root.innerHTML = "";
  const rows = [
    ["Profile", config.profile],
    ["Headline seeds", config.headline_seeds],
    ["Goal-directed seeds", config.goal_directed_seeds],
    ["Familiarity post repeats", config.familiarity_post_repeats],
    ["Lesion time", config.lesion_time],
  ];
  rows.forEach(([label, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    root.append(dt, dd);
  });
}

function chartBounds() {
  return { x: 66, y: 28, width: 522, height: 266 };
}

function clearSvg(svg) {
  while (svg.firstChild) {
    svg.removeChild(svg.firstChild);
  }
}

function svgNode(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function finiteExtent(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  if (min === max) {
    return [min - 1, max + 1];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function scaleLinear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (value) => {
    const t = (value - d0) / (d1 - d0);
    return r0 + (r1 - r0) * t;
  };
}

function drawAxes(svg, xLabel, yLabel, xTicks, yTicks) {
  const bounds = chartBounds();
  const axisColor = "rgba(24, 24, 24, 0.28)";

  svg.appendChild(svgNode("rect", {
    x: bounds.x,
    y: bounds.y,
    width: bounds.width,
    height: bounds.height,
    rx: 16,
    fill: "#ffffff",
    stroke: "rgba(0,0,0,0.08)",
  }));

  xTicks.forEach((tick) => {
    const line = svgNode("line", {
      x1: tick.x,
      x2: tick.x,
      y1: bounds.y + bounds.height,
      y2: bounds.y + bounds.height + 6,
      stroke: axisColor,
      "stroke-width": 1,
    });
    const label = svgNode("text", {
      x: tick.x,
      y: bounds.y + bounds.height + 22,
      fill: "#4a4f54",
      "font-size": 11,
      "text-anchor": "middle",
    });
    label.textContent = tick.label;
    svg.append(line, label);
  });

  yTicks.forEach((tick) => {
    const grid = svgNode("line", {
      x1: bounds.x,
      x2: bounds.x + bounds.width,
      y1: tick.y,
      y2: tick.y,
      stroke: "rgba(0,0,0,0.08)",
      "stroke-width": 1,
    });
    const line = svgNode("line", {
      x1: bounds.x - 6,
      x2: bounds.x,
      y1: tick.y,
      y2: tick.y,
      stroke: axisColor,
      "stroke-width": 1,
    });
    const label = svgNode("text", {
      x: bounds.x - 10,
      y: tick.y + 4,
      fill: "#4a4f54",
      "font-size": 11,
      "text-anchor": "end",
    });
    label.textContent = tick.label;
    svg.append(grid, line, label);
  });

  const xAxis = svgNode("line", {
    x1: bounds.x,
    x2: bounds.x + bounds.width,
    y1: bounds.y + bounds.height,
    y2: bounds.y + bounds.height,
    stroke: axisColor,
    "stroke-width": 1.2,
  });
  const yAxis = svgNode("line", {
    x1: bounds.x,
    x2: bounds.x,
    y1: bounds.y,
    y2: bounds.y + bounds.height,
    stroke: axisColor,
    "stroke-width": 1.2,
  });
  const xText = svgNode("text", {
    x: bounds.x + bounds.width / 2,
    y: bounds.y + bounds.height + 42,
    fill: "#161616",
    "font-size": 12,
    "text-anchor": "middle",
  });
  xText.textContent = xLabel;
  const yText = svgNode("text", {
    x: bounds.x - 48,
    y: bounds.y + bounds.height / 2,
    fill: "#161616",
    "font-size": 12,
    transform: `rotate(-90 ${bounds.x - 48} ${bounds.y + bounds.height / 2})`,
    "text-anchor": "middle",
  });
  yText.textContent = yLabel;
  svg.append(xAxis, yAxis, xText, yText);
}

function drawLegend(svg, labels) {
  const startX = 68;
  const startY = 18;
  labels.forEach((label, idx) => {
    const x = startX + idx * 160;
    svg.appendChild(svgNode("circle", {
      cx: x,
      cy: startY,
      r: 5,
      fill: modelColor(label),
    }));
    const text = svgNode("text", {
      x: x + 12,
      y: startY + 4,
      fill: "#161616",
      "font-size": 11,
    });
    text.textContent = label;
    svg.appendChild(text);
  });
}

function renderPlayScatter(data) {
  const svg = select("play-scatter");
  clearSvg(svg);
  const xMetric = select("play-x").value;
  const yMetric = select("play-y").value;
  const points = data.play.points.filter((row) => Number.isFinite(row[xMetric]) && Number.isFinite(row[yMetric]));
  const bounds = chartBounds();
  const xExtent = finiteExtent(points.map((row) => Number(row[xMetric])));
  const yExtent = finiteExtent(points.map((row) => Number(row[yMetric])));
  const xScale = scaleLinear(xExtent, [bounds.x, bounds.x + bounds.width]);
  const yScale = scaleLinear(yExtent, [bounds.y + bounds.height, bounds.y]);

  const xTicks = Array.from({ length: 5 }, (_, idx) => {
    const value = xExtent[0] + ((xExtent[1] - xExtent[0]) * idx) / 4;
    return { x: xScale(value), label: value.toFixed(1) };
  });
  const yTicks = Array.from({ length: 5 }, (_, idx) => {
    const value = yExtent[0] + ((yExtent[1] - yExtent[0]) * idx) / 4;
    return { y: yScale(value), label: value.toFixed(1) };
  });

  drawAxes(svg, metricLabel(data.play.metric_labels, xMetric), metricLabel(data.play.metric_labels, yMetric), xTicks, yTicks);
  drawLegend(svg, ["Recon", "Ipsundrum", "Ipsundrum+affect"]);

  points.forEach((row) => {
    const dot = svgNode("circle", {
      cx: xScale(Number(row[xMetric])),
      cy: yScale(Number(row[yMetric])),
      r: 5.2,
      fill: modelColor(row.model),
      opacity: 0.82,
    });
    const title = svgNode("title");
    title.textContent = `${row.model} seed ${row.seed}: ${xMetric}=${Number(row[xMetric]).toFixed(2)}, ${yMetric}=${Number(row[yMetric]).toFixed(2)}`;
    dot.appendChild(title);
    svg.appendChild(dot);
  });
}

function renderFamiliarityChart(data) {
  const svg = select("familiarity-chart");
  clearSvg(svg);
  const metric = select("familiarity-metric").value;
  const condition = select("familiarity-condition").value;
  const rows = data.familiarity.summary.filter((row) => row.familiarize_side === condition && Number.isFinite(row[metric]));
  const models = [...new Set(rows.map((row) => row.model))];
  const mornings = [...new Set(rows.map((row) => Number(row.morning_idx)))].sort((a, b) => a - b);
  const bounds = chartBounds();
  const values = rows.map((row) => Number(row[metric]));
  const yExtent = finiteExtent(values);
  const xScale = scaleLinear([Math.min(...mornings), Math.max(...mornings)], [bounds.x, bounds.x + bounds.width]);
  const yScale = scaleLinear(yExtent, [bounds.y + bounds.height, bounds.y]);

  const xTicks = mornings.map((value) => ({ x: xScale(value), label: String(value) }));
  const yTicks = Array.from({ length: 5 }, (_, idx) => {
    const value = yExtent[0] + ((yExtent[1] - yExtent[0]) * idx) / 4;
    return { y: yScale(value), label: value.toFixed(2) };
  });

  drawAxes(svg, "Morning index", metricLabel(data.familiarity.metric_labels, metric), xTicks, yTicks);
  drawLegend(svg, models);

  models.forEach((model) => {
    const series = rows.filter((row) => row.model === model).sort((a, b) => Number(a.morning_idx) - Number(b.morning_idx));
    const d = series
      .map((row, idx) => `${idx === 0 ? "M" : "L"} ${xScale(Number(row.morning_idx))} ${yScale(Number(row[metric]))}`)
      .join(" ");
    svg.appendChild(svgNode("path", {
      d,
      fill: "none",
      stroke: modelColor(model),
      "stroke-width": 3,
      "stroke-linecap": "round",
    }));
    series.forEach((row) => {
      svg.appendChild(svgNode("circle", {
        cx: xScale(Number(row.morning_idx)),
        cy: yScale(Number(row[metric])),
        r: 4.8,
        fill: modelColor(model),
      }));
    });
  });
}

function renderGoalChart(data) {
  const svg = select("goal-chart");
  clearSvg(svg);
  const task = select("goal-task").value;
  const metric = select("goal-metric").value;
  const rows = data.goal_directed.summary.filter((row) => row.task === task && Number.isFinite(row[metric]));
  const models = [...new Set(rows.map((row) => row.model))];
  const horizons = [...new Set(rows.map((row) => Number(row.horizon)))].sort((a, b) => a - b);
  const bounds = chartBounds();
  const values = rows.map((row) => Number(row[metric]));
  const yExtent = finiteExtent(values);
  const xScale = scaleLinear([Math.min(...horizons), Math.max(...horizons)], [bounds.x, bounds.x + bounds.width]);
  const yScale = scaleLinear(yExtent, [bounds.y + bounds.height, bounds.y]);

  const xTicks = horizons.filter((value) => value === 1 || value === 5 || value === 10 || value === 15 || value === 20).map((value) => ({
    x: xScale(value),
    label: String(value),
  }));
  const yTicks = Array.from({ length: 5 }, (_, idx) => {
    const value = yExtent[0] + ((yExtent[1] - yExtent[0]) * idx) / 4;
    return { y: yScale(value), label: value.toFixed(metric === "success_rate" ? 2 : 1) };
  });

  drawAxes(svg, "Planning horizon", metricLabel(data.goal_directed.metric_labels, metric), xTicks, yTicks);
  drawLegend(svg, models);

  models.forEach((model) => {
    const series = rows.filter((row) => row.model === model).sort((a, b) => Number(a.horizon) - Number(b.horizon));
    const d = series
      .map((row, idx) => `${idx === 0 ? "M" : "L"} ${xScale(Number(row.horizon))} ${yScale(Number(row[metric]))}`)
      .join(" ");
    svg.appendChild(svgNode("path", {
      d,
      fill: "none",
      stroke: modelColor(model),
      "stroke-width": 3,
    }));
  });
}

function zoomedExtent(values, minPad = 0.02) {
  const clean = values.filter((value) => Number.isFinite(value));
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const pad = Math.max((max - min) * 0.12, minPad);
  return [Math.max(0, min - pad), Math.min(1.02, max + pad)];
}

function renderLesionChart(data) {
  const svg = select("lesion-chart");
  clearSvg(svg);
  const model = select("lesion-model").value;
  const traces = data.lesion.mean_traces[model];
  const sham = traces.sham;
  const lesion = traces.lesion;
  const lesionT = Number(data.lesion.lesion_t);
  const windowStart = Math.max(0, lesionT - 1);
  const windowEnd = Math.min(sham.length - 1, lesionT + 16);
  const windowIndices = Array.from({ length: windowEnd - windowStart + 1 }, (_, offset) => windowStart + offset);
  const zoomValues = windowIndices.flatMap((idx) => [Number(sham[idx]), Number(lesion[idx])]);
  const bounds = chartBounds();
  const xScale = scaleLinear([windowStart, windowEnd], [bounds.x, bounds.x + bounds.width]);
  const yExtent = zoomedExtent(zoomValues);
  const yScale = scaleLinear(yExtent, [bounds.y + bounds.height, bounds.y]);
  const xTickValues = Array.from(new Set([
    windowStart,
    lesionT,
    lesionT + 1,
    lesionT + 3,
    lesionT + 7,
    windowEnd,
  ].filter((value) => value >= windowStart && value <= windowEnd))).sort((a, b) => a - b);
  const xTicks = xTickValues.map((value) => ({
    x: xScale(value),
    label: String(value),
  }));
  const yTicks = Array.from({ length: 5 }, (_, idx) => {
    const value = yExtent[0] + ((yExtent[1] - yExtent[0]) * idx) / 4;
    return {
      y: yScale(value),
      label: value.toFixed(2),
    };
  });

  drawAxes(svg, "Time step", "Mean Ns", xTicks, yTicks);
  const modelText = svgNode("text", {
    x: 68,
    y: 18,
    fill: "#161616",
    "font-size": 11,
  });
  modelText.textContent = model;
  svg.appendChild(modelText);
  const zoomText = svgNode("text", {
    x: bounds.x + bounds.width,
    y: 18,
    fill: "#4a4f54",
    "font-size": 11,
    "text-anchor": "end",
  });
  zoomText.textContent = `Zoom ${windowStart}-${windowEnd}`;
  svg.appendChild(zoomText);
  svg.appendChild(svgNode("line", {
    x1: 154,
    x2: 178,
    y1: 14,
    y2: 14,
    stroke: "#f4efe6",
    "stroke-width": 2.8,
  }));
  const shamText = svgNode("text", {
    x: 184,
    y: 18,
    fill: "#161616",
    "font-size": 11,
  });
  shamText.textContent = "Sham";
  svg.appendChild(shamText);
  svg.appendChild(svgNode("line", {
    x1: 236,
    x2: 260,
    y1: 14,
    y2: 14,
    stroke: "#de684a",
    "stroke-width": 2.8,
    "stroke-dasharray": "8 6",
  }));
  const lesionText = svgNode("text", {
    x: 266,
    y: 18,
    fill: "#161616",
    "font-size": 11,
  });
  lesionText.textContent = "Lesion";
  svg.appendChild(lesionText);
  const lesionBand = svgNode("rect", {
    x: xScale(lesionT),
    y: bounds.y,
    width: xScale(windowEnd) - xScale(lesionT),
    height: bounds.height,
    fill: "rgba(240, 162, 69, 0.08)",
  });
  svg.appendChild(lesionBand);
  const lesionLine = svgNode("line", {
    x1: xScale(lesionT),
    x2: xScale(lesionT),
    y1: bounds.y,
    y2: bounds.y + bounds.height,
    stroke: "#ffcf96",
    "stroke-width": 1.4,
    "stroke-dasharray": "6 4",
  });
  svg.appendChild(lesionLine);

  const makePath = (series) => windowIndices
    .map((idx, pointIdx) => `${pointIdx === 0 ? "M" : "L"} ${xScale(idx)} ${yScale(Number(series[idx]))}`)
    .join(" ");
  svg.appendChild(svgNode("path", {
    d: makePath(sham),
    fill: "none",
    stroke: "#f4efe6",
    "stroke-width": 2.8,
  }));
  svg.appendChild(svgNode("path", {
    d: makePath(lesion),
    fill: "none",
    stroke: "#de684a",
    "stroke-width": 2.8,
    "stroke-dasharray": "8 6",
  }));
}

function installRevealObserver() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
      }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
}

async function main() {
  const response = await fetch("static/data/site-data.json");
  const data = await response.json();

  renderHeadlineMetrics(data.headline_metrics);
  renderGallery(data.gallery);
  renderConfig(data.config);

  setLink("paper-link", data.links.paper_pdf);
  setLink("paper-link-secondary", data.links.paper_pdf);
  setLink("repo-link-top", data.links.repo);
  setLink("repo-link", data.links.repo);
  setLink("colab-link", data.links.colab);
  setLink("colab-link-secondary", data.links.colab);

  setImage("hero-media", data.assets.hero);
  setImage("gridworld-media", data.assets.gridworld);
  setImage("lesion-media", data.assets.lesion);
  setImage("stage-strip", data.assets.stage_strip);
  setImage("play-summary-media", data.assets.play_summary);
  setImage("play-trajectories-media", data.assets.play_trajectories);
  setImage("play-heatmaps-media", data.assets.play_heatmaps);
  setImage("familiarity-summary-media", data.assets.familiarity_summary);
  setImage("familiarity-supp-media", data.assets.familiarity_supp);
  setImage("lesion-summary-media", data.assets.lesion_summary);
  setImage("pain-tail-media", data.assets.pain_tail);
  setImage("qualiaphilia-media", data.assets.qualiaphilia);

  populateSelect(
    select("play-x"),
    Object.entries(data.play.metric_labels).map(([value, label]) => ({ value, label })),
    data.play.default_x,
  );
  populateSelect(
    select("play-y"),
    Object.entries(data.play.metric_labels).map(([value, label]) => ({ value, label })),
    data.play.default_y,
  );
  populateSelect(
    select("familiarity-metric"),
    Object.entries(data.familiarity.metric_labels).map(([value, label]) => ({ value, label })),
    data.familiarity.default_metric,
  );
  populateSelect(
    select("familiarity-condition"),
    ["scenic", "dull", "both", "none"].map((value) => ({ value, label: value })),
    data.familiarity.default_condition,
  );
  populateSelect(
    select("goal-task"),
    ["gridworld", "corridor"].map((value) => ({ value, label: value })),
    data.goal_directed.default_task,
  );
  populateSelect(
    select("goal-metric"),
    Object.entries(data.goal_directed.metric_labels).map(([value, label]) => ({ value, label })),
    data.goal_directed.default_metric,
  );
  populateSelect(
    select("lesion-model"),
    Object.keys(data.lesion.mean_traces).map((value) => ({ value, label: value })),
    "Ipsundrum+affect",
  );

  const rerender = () => {
    renderPlayScatter(data);
    renderFamiliarityChart(data);
    renderGoalChart(data);
    renderLesionChart(data);
  };

  ["play-x", "play-y", "familiarity-metric", "familiarity-condition", "goal-task", "goal-metric", "lesion-model"]
    .forEach((id) => select(id).addEventListener("change", rerender));

  rerender();
  installRevealObserver();
}

main().catch((error) => {
  console.error(error);
});
