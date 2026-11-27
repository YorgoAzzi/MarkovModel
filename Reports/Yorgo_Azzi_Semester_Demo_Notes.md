# Movement to sound: semester demo

**Presenter:** Yorgo Azzi  
**Target length:** 7–8 minutes  
**Deck:** `Yorgo_Azzi_Movement_to_Sound_Semester_Demo_Screenshots.pptx`

## Before presenting

1. Replace the labeled space on slide 2 with your own colored-marker detection screenshot.
2. Connect the webcam, confirm MIDI output, and use even lighting.
3. Start the app from the project root:

   ```powershell
   python Code/hand_performance.py --performance-mode conductor --tempo 60 --max-tempo 140 --key C --mode major
   ```

   If the beat detector needs a stronger threshold, add `--conductor-acceleration 10`.

## Talk and demo sequence

| Time | Slide | Main point |
|---|---|---|
| 0:00–0:45 | 1 | Show the real app screenshot. The Markov generator became a hand-controlled performance system that produces sound and visual output. |
| 0:45–2:20 | 2 | Show the generated score, then explain the early OpenCV colored-marker prototype and its height-to-volume mapping. Realtime MIDI made continuous control possible. |
| 2:20–3:55 | 3 | Compare the piano-roll capture with the later camera-effects and painting capture. Explain OpenCV camera capture, MediaPipe hand tracking, the Markov engine, and MIDI/visual output. |
| 3:55–4:40 | 4 | Press **1**. Demonstrate Full SATB: gestures influence music that the model generates in four voices. |
| 4:40–5:30 | 4 | Press **2**. Demonstrate Soprano Performance: play the soprano with the right hand while the model supplies the lower voices. |
| 5:30–6:45 | 4 | Press **3**. Demonstrate Conductor: a downward stroke triggers a beat, right fingers choose chords, and gestures shape the melody between beats. |
| 6:45–7:30 | 4 | Press **P** for painting and **V** for camera effects. Close by showing the hand-distance light and the audible/visual response. |

The slides contain only real app screenshots and an excerpt rendered from the project's generated score. The marker image on slide 2 is deliberately left for you to add.
