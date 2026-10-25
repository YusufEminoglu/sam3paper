# Phantom Capacity in Urban Pedestrian Networks

Code and derived data for the manuscript *Phantom Capacity in Urban Pedestrian Networks: An AI-Assisted Approach to Functional Walkability Using Street View Imagery and SAM 3* (Eminoğlu, Erdin, Şanlı), submitted to *Transactions in GIS*.

The repository contains the full measurement pipeline, the validation and audit scripts, the analysis and figure scripts, and the derived tables that the manuscript reports. Raw Street View frames and the municipal GIS database are not included; the reasons are given below.

## What the method does

A pedestrian width recorded in a plan is not the width that can be walked. For every street node the pipeline:

1. reconstructs the street corridor from municipal building footprints and road centrelines, separating the carriageway from the pedestrian-intended zone (`02_pipeline_v2/11_corridor_width.py`);
2. selects the two most street-perpendicular Street View views per node (`12_build_manifest.py`, `12b_manifest_bestviews.py`, `13_view_coverage.py`);
3. segments obstacles with SAM 3, prompted by text concepts (`20_sam3_segment.py`);
4. projects the ground-contact line of every mask onto the ground plane with an explicit camera model, anchors the camera position on GIS facades, and calibrates the camera height from opposite facades (`geom.py`, `anchor.py`, `42_calibrate_h.py`);
5. merges the views of a node into a raster of the pedestrian-intended zone and measures the bottleneck of the continuous free path, which is the Effective Clear Width (ECW), together with phantom capacity and Monte Carlo uncertainty (`30_ecw_v2.py`);
6. validates the geometry against GIS, compares SAM 3 with a fixed-vocabulary model (SegFormer-B5, ADE20K), audits segmentation instances, and models nodes nested in streets (`03_validation/`, `04_analysis/`).

The analysed sample has 557 nodes in 251 streets in Konak, İzmir.

## Repository layout

```text
00_inputs/
  konak_gsv_candidates.csv      # 876 candidate nodes of the Konak sample (node_id, lat, lon, hex_id)
  legacy_gsv_nodes.csv          # the 63 nodes of the first-version pilot, with capture year
02_pipeline_v2/                 # corridor geometry, view selection, SAM 3 segmentation, ECW, calibration
  *.py                          # pipeline scripts (run from this folder)
  *.csv                         # derived outputs: corridor_nodes, manifest_run, view_coverage, seg scores, calib,
                                #   ecw_v2, ecw_sf, geomval, anchor_cons, panorama provenance and capture dates
03_validation/                  # model comparison, instance audit, audit tool
  out_final/*.csv               # agreement and audit summaries
  audit/*.csv                   # filled audit sheets (images are not redistributed)
04_analysis/                    # node and street analysis, sensitivity, tables, figures
  out_final/*.csv               # nodes_R1, streets_R1, class summaries, mixed model, sensitivity, selection test
  *.py                          # analysis and figure scripts
legacy_v1/                      # the five scrubbed scripts of the first version, and its 63-node result table
```

The derived tables are sufficient to rerun `04_analysis/` and reproduce the reported statistics and figures without raw imagery. Regenerating the segmentation or the geometry requires the inputs listed below.

## What is not in this repository

**Street View frames.** Google's terms do not permit redistribution, so no frame is stored here, and the audit sheets are provided as verdicts rather than images. Every frame used in the study is identified in `02_pipeline_v2/manifest_run.csv` by node, side, heading, pitch, panorama identifier and capture date. Anyone with a Street View API key can request the same frames again. The legacy pilot was re-retrieved for this revision with panoramas fixed by identifier and screened by capture date; the provenance is in `tg_images_R1_provenance.csv` and `tg_pano_dates.csv`.

**Municipal GIS database.** The road and building layers come from the Konak smart-city database and are not redistributed. Derived per-node corridor geometry is provided in `corridor_nodes.csv`. To rebuild it, set `PC_MUNICIPAL_GPKG` to the municipal GeoPackage and run `10_explore_gpkg.py` and `11_corridor_width.py`.

**Segmentation masks and the full output archive.** The 2,349 mask files (18 MB), the per-image segmentation scores and the complete derived tables are deposited on figshare under CC BY 4.0. The DOI is reported in the Data Availability Statement of the manuscript.

**Audit images.** `50_audit_sheets.py` and `51_audit_tool.py` regenerate the sheets and the HTML audit tool from the frames and masks; the tool is not committed because it embeds the photographs.

## Reproducing the analysis

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # torch cu128 build for NVIDIA GPUs

export HF_TOKEN=...                                  # gated access to facebook/sam3
export PC_MUNICIPAL_GPKG=/path/municipal.gpkg
export PC_SCS_IMAGES=/path/konak_frames
export PC_LEGACY_IMAGES=/path/legacy_frames

cd 02_pipeline_v2
python test_geom.py                                  # geometry unit tests
python 10_explore_gpkg.py && python 11_corridor_width.py && python 14_street_units.py
python 13_view_coverage.py && python 12b_manifest_bestviews.py
python 20_sam3_segment.py manifest_run.csv seg_v2     # about 16 s per image on an NVIDIA T1000, fp32
python 42_calibrate_h.py seg_v2 manifest_run.csv calib.csv
python 30_ecw_v2.py seg_v2 manifest_run.csv ecw_v2.csv 30 0 calib.csv
cd ../04_analysis
python 50_analysis.py ../02_pipeline_v2/ecw_v2.csv ../02_pipeline_v2/corridor_nodes.csv out_final
```

The scripts in `04_analysis/` read their inputs from `02_pipeline_v2/` and `04_analysis/out_final/`, so the repository can be used in place. Figure output goes to `figures/` unless `PC_FIG_DIR` is set.

Hardware note: SAM 3 must run in fp32 on an NVIDIA T1000 (8 GB, Turing); fp16 was four times slower. Sharing one image embedding across the text prompts of an image reproduces the single-prompt scores exactly and cuts the run time from about 320 s to about 16 s per image.

## Validation and audit

The geometry was validated against GIS facades only. No field survey was carried out. The segmentation audit was desk-based, performed by one rater with an AI-assisted display, on 15 images and 143 instances; precision was 90.2% (Wilson 95% interval 84.2 to 94.1). These limits are stated in the manuscript.

## Licence and citation

Code is released under the MIT Licence (`LICENSE`). Derived data in this repository are released under CC BY 4.0 (`DATA_LICENSE.md`). If you use the code or the data, please cite the article and this repository; `CITATION.cff` gives the metadata.

## Contact

Yusuf Eminoğlu, Department of City and Regional Planning, Dokuz Eylül University, İzmir, Türkiye.
yusuf.eminoglu@deu.edu.tr, ORCID 0009-0005-6000-2934.
