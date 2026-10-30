# Animations and downloads

[← Documentation](README.md) · [View the animations in the README](../README.md)

The RFdiffusion3 phosphotriesterase movie is the README highlight. All three
animations retain their full timelines, title cards, credits, and pauses.

## Original media

Both source MP4s are preserved unchanged at 1920 × 1080. The RFdiffusion2
inline GIF and cropped MP4 remove the black bars; the original remains available
below. The RFdiffusion3 supplementary GIF is unchanged.

| Animation | Duration | Original download | README animation |
| :--- | :--- | :--- | :--- |
| **RFdiffusion3 highlight** — phosphotriesterase theozyme and scaffold-generation trajectory | 58.17 s | [MP4](assets/rfdiffusion3-highlight.mp4?raw=true) | [Looping GIF](assets/rfdiffusion3-highlight.gif) |
| **RFdiffusion2** — scaffold generation | 5.77 s | [Original MP4](assets/rfdiffusion2-scaffold-generation-original.mp4?raw=true) | [Cropped GIF](assets/rfdiffusion2-scaffold-generation.gif) · [Cropped MP4](assets/rfdiffusion2-scaffold-generation.mp4?raw=true) |
| **RFdiffusion3 supplementary** — active-site and scaffold detail | 11.00 s | [Original GIF](assets/rfdiffusion3-supplementary.gif) | Same original GIF |

These movies illustrate computational models and design trajectories. The
[experimental datasets](data-and-reproduction.md) provide the associated
measurements; the [citation guide](citations.md) identifies the publications.

## Playback

GIFs loop continuously, subject to readers’ [GitHub motion settings](https://docs.github.com/en/account-and-profile/how-tos/account-settings/managing-accessibility-settings#managing-motion).

Use the original MP4 downloads for source-resolution playback, or the cropped
RFdiffusion2 MP4 without black bars. GIFs are sized for inline display.

## Preservation and conversion

Every source frame is retained in order, with no temporal trimming. The cropped
RFdiffusion2 MP4 preserves all 173 frame timestamps and its 5.772433 s duration.
Its spatial crop is `1920:800:0:140` (width:height:x:y): 140 pixels removed from
the top and bottom, with the full width and all visible content retained.
The crop was checked across all source frames; the GIF scales it to 720 × 300.

GIF timing uses centiseconds: 58.166667 s and 5.772433 s become 58.17 s and
5.77 s. The supplementary GIF retains its original variable delays.
Conversions use a 128-color palette with ordered dithering and lossless
`gifsicle -O3` optimization. RFdiffusion2 also uses `-Okeep-empty` to retain
repeated frames. Every optimized frame and individual delay was verified
against its initial GIF; RFdiffusion2 delays also match the uncropped GIF.

| File | Source frames retained | Display dimensions |
| :--- | ---: | :--- |
| `rfdiffusion3-highlight.gif` | 1,396 | 960 × 540 |
| `rfdiffusion2-scaffold-generation.gif` | 173 | 720 × 300 |
| `rfdiffusion2-scaffold-generation.mp4` (cropped) | 173 | 1920 × 800 |
| `rfdiffusion3-supplementary.gif` | 128 | 1200 × 900 |

Original-file SHA-256 checksums:

```text
9bf3c054e1dd5af86129b84a24febc141c003081f7450fa649a06ff6596b6574  rfdiffusion3-highlight.mp4
c0da495f48dec8c140596ee8ad77b70725cf2a521c3a0e26e85012b6f94102ad  rfdiffusion2-scaffold-generation-original.mp4
d0ebe34ef94fb40fb870e6f035166cd107a1a8695096babc5d373bc27500f1c3  rfdiffusion3-supplementary.gif
```

Cropped RFdiffusion2 SHA-256 checksums:

```text
77c23882fe5fcf4a3174a607a8ff913d968c3e5da02ed8f81b33c7bb76ddd257  rfdiffusion2-scaffold-generation.mp4
5e69ea109408a4189b4ecc041a5d0622d0949d14ba21a8008953f14c889ce126  rfdiffusion2-scaffold-generation.gif
```
