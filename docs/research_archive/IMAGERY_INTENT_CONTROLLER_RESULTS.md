# Imagery-Only Controller Results

The controller was trained only on motor-imagery data. It used 68 available development subjects, 17 complete controller-validation subjects, and 22 completely untouched final patients. Subject 109 was unavailable in the cached dataset; controller-validation subject 48 had only one usable Run 12 trial and was excluded from controller selection. Neither issue affected the final patient cohort.

Each final patient contributed 12 labeled trials per imagery direction and 18 rest trials from Runs 4/8 for calibration. Run 12 remained held out. The policy was selected using only the controller-validation cohort.

| Operating point | Final false activation | Final missed intent | Final direction BA |
| --- | ---: | ---: | ---: |
| Conservative | 19.6% | 54.0% | 70.9% |
| Balanced | 28.1% | 36.6% | 70.9% |
| Responsive | 28.1% | 36.6% | 70.9% |

The conservative policy met its 10% false-activation design target on controller-validation patients (8.8%) but did not generalize to final patients (19.6%). This is a failed safety generalization, not a deployable result. The direction model is materially better than chance once an imagery trial has been accepted, but the rest-versus-intent gate remains unstable across patients and runs.

The saved figures are [controller_tradeoff_frontier.png](/Users/LiamKennings_1/Documents/BME%20473/artifacts/imagery_intent_controller/controller_tradeoff_frontier.png), [final_operating_points.png](/Users/LiamKennings_1/Documents/BME%20473/artifacts/imagery_intent_controller/final_operating_points.png), and [final_patient_tradeoff.png](/Users/LiamKennings_1/Documents/BME%20473/artifacts/imagery_intent_controller/final_patient_tradeoff.png). The full machine-readable output is [imagery_intent_controller_metrics.json](/Users/LiamKennings_1/Documents/BME%20473/artifacts/imagery_intent_controller/imagery_intent_controller_metrics.json).

The next experiment should not lower the threshold. It should collect or simulate sustained idle data, update the rest-score distribution online, require consecutive evidence windows before emitting a command, and apply a short refractory period. That continuous-time experiment needs a separate raw-window protocol because EEGMMI’s cue-locked labels cannot measure false commands per minute during natural idle behavior.
