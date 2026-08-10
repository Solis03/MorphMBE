import fs from "node:fs/promises";
import path from "node:path";

import { Presentation, PresentationFile } from "@oai/artifact-tool";


const packageRoot = path.resolve(process.argv[2]);
const editableRoot = path.join(packageRoot, "editable");
const previewRoot = path.join(editableRoot, "previews");

const figures = [
  {
    stem: "Figure_1_AutoRHEED_M22_overview",
    width: 672,
    height: 427.2,
    alt: "Figure 1: AutoRHEED M20 and M22c framework overview",
  },
  {
    stem: "Figure_2_M20_M22_model_and_validation",
    width: 672,
    height: 590.4,
    alt: "Figure 2: M20 and M22c model and leakage-controlled validation",
  },
  {
    stem: "Figure_3_M22_selected_results_and_Sq",
    width: 672,
    height: 801.6,
    alt: "Figure 3: selected M22 AFM results, Sq scatter, and ordered Sq profile",
  },
  {
    stem: "Figure_4_M22_full_cohort_atlas",
    width: 816,
    height: 2496,
    alt: "Figure 4: full 27-growth M22 RHEED-to-AFM atlas",
  },
];

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(editableRoot, { recursive: true });
  await fs.mkdir(previewRoot, { recursive: true });
  const manifest = [];
  for (const spec of figures) {
    const svgPath = path.join(packageRoot, "figures", `${spec.stem}.svg`);
    const svg = await fs.readFile(svgPath);
    const svgBytes = svg.buffer.slice(
      svg.byteOffset,
      svg.byteOffset + svg.byteLength,
    );
    const presentation = Presentation.create({
      slideSize: { width: spec.width, height: spec.height },
    });
    const slide = presentation.slides.add();
    slide.background.fill = "white";
    slide.images.add({
      blob: svgBytes,
      contentType: "image/svg+xml",
      alt: spec.alt,
      fit: "contain",
      position: {
        left: 0,
        top: 0,
        width: spec.width,
        height: spec.height,
      },
    });
    slide.speakerNotes.textFrame.setText(
      `[Sources]\n- ${svgPath}\n- M22 strict outer-LOO data: reports/rheed_m22_dense_mid/20260809_m22_inclusive_v1/full27_loo/\n- AFM/RHEED panels: frozen repository-derived scientific assets\n[/Sources]`,
    );
    const outputPath = path.join(editableRoot, `${spec.stem}_editable.pptx`);
    const previewPath = path.join(previewRoot, `${spec.stem}_artifact_preview.png`);
    const layoutPath = path.join(previewRoot, `${spec.stem}.layout.json`);
    await writeBlob(
      previewPath,
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(layoutPath, await layout.text());
    const deck = await PresentationFile.exportPptx(presentation);
    await deck.save(outputPath);
    manifest.push({
      figure: spec.stem,
      pptx: outputPath,
      source_svg: svgPath,
      slide_size_px: [spec.width, spec.height],
      preview: previewPath,
      layout: layoutPath,
    });
  }
  await fs.writeFile(
    path.join(editableRoot, "editable_figure_manifest.json"),
    `${JSON.stringify({ status: "built", figures: manifest }, null, 2)}\n`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
