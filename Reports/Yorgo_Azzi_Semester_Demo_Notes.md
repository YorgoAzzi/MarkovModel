# Movement to sound: semester demo

**Presenter:** Yorgo Azzi  
**Target length:** 7-8 minutes  
**Deck:** `Yorgo_Azzi_Movement_to_Sound_Semester_Demo_Revised.pptx` (4 slides)

## Before the presentation

1. Connect the webcam and check that the MIDI output is audible.
2. Use even lighting and keep both hands in the camera frame.
3. Start the app from the project root:

   ```powershell
   python Code/hand_performance.py --performance-mode conductor --tempo 60 --max-tempo 140 --key C --mode major
   ```

   If the tested setup needs a stronger beat threshold, add `--conductor-acceleration 10`.
4. Open Magenta RealTime on a compatible machine only if you plan to demonstrate its optional audio output. The main MIDI demo works without it.

## Talk and demo sequence

| Time | Slide | What to do and say |
|---|---|---|
| 0:00-0:40 | 1 | Explain the objective: hand movement controls when symbolic music is generated and how it sounds. |
| 0:40-2:15 | 2 | Compare the generated score with the current hand-controlled app. Explain the realtime MIDI engine and how MediaPipe landmarks feed the controls. The app image is an interface preview with an illustrative camera frame; show the live camera during the demo. |
| 2:15-3:45 | 3 | Walk through the evolution: offline Markov score, realtime MIDI, OpenCV ribbon, MediaPipe hands, and one live app. Explain the three performance modes. Mention optional Magenta routing briefly. |
| 3:45-4:45 | 4 | Show both hands on camera. Hold a left fist and bring both hands down to trigger a beat. Change the right-hand finger pattern to choose another chord degree. |
| 4:45-6:00 | 4 | Move the right hand vertically for soprano notes. Change left-hand fingers to vary melodic jumps. Show that the melody reacts between conducted beats. |
| 6:00-6:50 | 4 | Press **P** to show the painting and **V** for camera particles. The painting color follows the current chord. |
| 6:50-7:30 | 4 | Close with the result: the performer commands beat timing while gestures shape melody and harmony. |

If the camera misses a gesture, slow the hand movement and keep fingers visible. If Magenta is unavailable, omit it from the live sequence; the direct MIDI and visual demonstration still shows the core project.
