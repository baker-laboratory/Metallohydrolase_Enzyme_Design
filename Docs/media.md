# Animations and downloads

[← Documentation](README.md) · [View the animations in the README](../README.md)

The RFdiffusion3 phosphotriesterase movie is the README highlight. All three
animations retain their full timelines, title cards, credits, and pauses.

## Original media

Both source MP4s and the original supplementary GIF are preserved unchanged.
The inline RFdiffusion2 and supplementary RFdiffusion3 animations use safe
spatial crops and a shared display height, retaining their natural proportions.

| Animation | Duration | Original download | README animation |
| :--- | :--- | :--- | :--- |
| **RFdiffusion3 highlight** — phosphotriesterase theozyme and scaffold-generation trajectory | 58.17 s | [MP4](assets/rfdiffusion3-highlight.mp4?raw=true) | [Looping GIF](assets/rfdiffusion3-highlight.gif) |
| **RFdiffusion2** — scaffold generation | 5.77 s | [Original MP4](assets/rfdiffusion2-scaffold-generation-original.mp4?raw=true) | [Cropped GIF](assets/rfdiffusion2-scaffold-generation.gif) · [Cropped MP4](assets/rfdiffusion2-scaffold-generation.mp4?raw=true) |
| **RFdiffusion3 supplementary** — active-site and scaffold detail | 11.00 s | [Original GIF](assets/rfdiffusion3-supplementary-original.gif) | [Cropped GIF](assets/rfdiffusion3-supplementary.gif) |

These movies illustrate computational models and design trajectories. The
[experimental datasets](data-and-reproduction.md) provide the associated
measurements; the [citation guide](citations.md) identifies the publications.

## Playback

GIFs loop continuously, subject to readers’ [GitHub motion settings](https://docs.github.com/en/account-and-profile/how-tos/account-settings/managing-accessibility-settings#managing-motion).

Use the original MP4 downloads for source-resolution playback, or the cropped
RFdiffusion2 MP4 without black bars. GIFs are sized for inline display.

## Preservation and conversion

Every source frame is retained in order, with no temporal trimming. The
RFdiffusion2 MP4 preserves all 173 timestamps and its 5.772433 s duration.
Crops were checked against visible content across every source frame:

| Animation | Source crop (`width:height:x:y`) | Margins removed |
| :--- | :--- | :--- |
| RFdiffusion2 | `1760:798:160:141` | 160 px of blank space at left; 141 px black bars at top and bottom |
| RFdiffusion3 supplementary | `1200:848:0:6` | 6 px of white space at top; 46 px at bottom |

RFdiffusion2 content already reaches the top and bottom, so further vertical
cropping would remove molecular content. Both GIFs are scaled to 480 px tall,
with natural widths rounded to whole pixels and no added canvas padding.

GIF timing uses centiseconds: 58.166667 s and 5.772433 s become 58.17 s and
5.77 s. The supplementary GIF preserves every original variable delay,
including its pauses. Conversions use a 128-color ordered palette and lossless
`gifsicle -O3` optimization (`-Okeep-empty` for the two cropped GIFs).
Every optimized frame and individual delay was checked against its initial GIF;
cropped GIF delays also match their preceding versions or original source.

| File | Source frames retained | Display dimensions |
| :--- | ---: | :--- |
| `rfdiffusion3-highlight.gif` | 1,396 | 960 × 540 |
| `rfdiffusion2-scaffold-generation.gif` | 173 | 1059 × 480 |
| `rfdiffusion2-scaffold-generation.mp4` (cropped) | 173 | 1760 × 798 |
| `rfdiffusion3-supplementary.gif` | 128 | 679 × 480 |

Original-file SHA-256 checksums:

```text
9bf3c054e1dd5af86129b84a24febc141c003081f7450fa649a06ff6596b6574  rfdiffusion3-highlight.mp4
c0da495f48dec8c140596ee8ad77b70725cf2a521c3a0e26e85012b6f94102ad  rfdiffusion2-scaffold-generation-original.mp4
d0ebe34ef94fb40fb870e6f035166cd107a1a8695096babc5d373bc27500f1c3  rfdiffusion3-supplementary-original.gif
```

Cropped-file SHA-256 checksums:

```text
53f3c5fae29948a3b570a554d41410df8ed072988f093ccd9c0f34f986e402bd  rfdiffusion2-scaffold-generation.mp4
04457dd0d1a3a54b4ace700cd9fae4d946a7cfaab9904365cdab435189ec5982  rfdiffusion2-scaffold-generation.gif
331f79339cdf02c9f9cd2f61f01071047261de72534434dc960b56b1f5fd8fa4  rfdiffusion3-supplementary.gif
```
