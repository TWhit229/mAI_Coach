# HOW TO RUN Video Cut Tool
cd /path/to/Video-Cut-Tool

python .\auto-cut-video.py --speed 0.5 --pad_ms 120 --out_dir ./cut-videos

# Tool Instructions
1. Select video to process

Controls (new + recap)

Space = mark bottom

U = undo last mark

P or K = pause/play

← / → = rewind/forward 0.2 s (also pauses so you can fine-tune)

J / L = rewind/forward 0.5 s

, / . = single-frame step back/forward (works while paused)

[ / ] = slower/faster playback

Q or Esc = finish and export clips

2. When finished with all cuts, press ESC or Q to exit 
3. Cut videos will show up in ./cut-videos folder