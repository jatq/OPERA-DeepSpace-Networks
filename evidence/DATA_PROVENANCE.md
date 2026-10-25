# External Timing Data Provenance

The external datasets supply contact timing only. Capacity, tasks, prices,
failures, forecasts, authorization parameters, and workload overlays are
controlled synthetic inputs. The replay results therefore do not estimate
operational DSN throughput.

## SOHO DSN schedule archive

- Source: `https://soho.nascom.nasa.gov/soc/dsn/dsn_merged.dat`
- Access date: September 30, 2026
- Frozen source SHA-256: `c4f8561a8427e7477e26f7dd5991404e85596d4a6007c7511b46a4ca7d8c8142`
- Records used: 5,872 scheduling records from July 31, 2017 to July 18, 2022
- Temporal partition: 31 nonoverlapping segments
- Fields used: start time, end time, and DSS antenna identifier

## InSight RISE Tracking Data Message archive

- DOI: `10.17189/1517569`
- Source file: `nsyt_maro_2021_001_2022_147.tdm`
- Frozen source SHA-256: `262d8a99677ccb373e37c9b9f1c6c795ba6538634316efd3ba173a0b454f3803`
- Doppler records parsed: 2,976
- Gap rule for observation support: 300 s
- Resulting observation intervals: 105
- Contiguous nonreused replay segments: 26

## InSight TNF product intervals

- Source: NASA Planetary Data System collection API, accessed through the
  RISE archive identified by DOI `10.17189/1517569`
- Frozen API response SHA-256: `8e699bcd13e4121de8e40bd8e14e79e4ae54c739e7084e8350b775d898cd71f5`
- Product labels parsed: 605
- Complete single-station intervals: 604
- Exclusion: one `mmmv1` multi-station product without a unique DSS identifier
- Duration range: 43.4--634.4 min; median: 170.4 min

## Redistribution

The original external files are not redistributed in this archive. The frozen
source hashes above identify the exact public snapshots used, and the counts and
segmentation rules allow the timing partitions to be reconstructed from the
cited public sources. The controlled synthetic overlays are the variables used
in the paper's replay experiments.
