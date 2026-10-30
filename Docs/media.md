# Animations and downloads

[← Documentation](README.md) · [View the animations in the README](../README.md)

The RFdiffusion3 phosphotriesterase movie is the README highlight. All three
animations retain their full timelines, title cards, credits, and pauses.

## Original media

The two MP4s retain their original 1920 × 1080 resolution and have not been
re-encoded. The supplementary GIF is also preserved unchanged.

| Animation | Duration | Original download | README animation |
| :--- | :--- | :--- | :--- |
| **RFdiffusion3 highlight** — phosphotriesterase theozyme and scaffold-generation trajectory | 58.17 s | [MP4](assets/rfdiffusion3-highlight.mp4?raw=true) | [Looping GIF](assets/rfdiffusion3-highlight.gif) |
| **RFdiffusion2** — scaffold generation | 5.77 s | [MP4](assets/rfdiffusion2-scaffold-generation.mp4?raw=true) | [Looping GIF](assets/rfdiffusion2-scaffold-generation.gif) |
| **RFdiffusion3 supplementary** — active-site and scaffold detail | 11.00 s | [Original GIF](assets/rfdiffusion3-supplementary.gif) | Same original GIF |

These movies illustrate computational models and design trajectories. The
[experimental datasets](data-and-reproduction.md) provide the associated
measurements; the [citation guide](citations.md) identifies the publications.

## Playback

GIFs loop continuously, subject to readers’ [GitHub motion settings](https://docs.github.com/en/account-and-profile/how-tos/account-settings/managing-accessibility-settings#managing-motion).

Use the MP4 downloads for full-resolution playback; the GIFs are sized for
inline display.

## Preservation and conversion

Every source frame is retained in order, without cropping. GIF timing uses
centiseconds: 58.166667 s and 5.772433 s become 58.17 s and 5.77 s. The
supplementary GIF retains its original variable delays.

The conversions use a 128-color palette with ordered dithering. The highlight
also uses lossless `gifsicle -O3` optimization; decoded frames and individual
delays were verified against the initial GIF.

| File | Source frames retained | Display dimensions |
| :--- | ---: | :--- |
| `rfdiffusion3-highlight.gif` | 1,396 | 960 × 540 |
| `rfdiffusion2-scaffold-generation.gif` | 173 | 720 × 405 |
| `rfdiffusion3-supplementary.gif` | 128 | 1200 × 900 |

Original-file SHA-256 checksums:

```text
9bf3c054e1dd5af86129b84a24febc141c003081f7450fa649a06ff6596b6574  rfdiffusion3-highlight.mp4
c0da495f48dec8c140596ee8ad77b70725cf2a521c3a0e26e85012b6f94102ad  rfdiffusion2-scaffold-generation.mp4
d0ebe34ef94fb40fb870e6f035166cd107a1a8695096babc5d373bc27500f1c3  rfdiffusion3-supplementary.gif
```
