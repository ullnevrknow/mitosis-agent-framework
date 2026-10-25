# Experiment results

4 scenarios x 20 seeds, settings {'spot_check_every': 5, 'manager_miss_rate': 0.0, 'audit_on_flag': True}.
Offline proof model: scripted Executor and Memory Manager.

| Metric (average per 6-episode series) | No protection | Filter only | Memory Manager only | Full system |
|---|---:|---:|---:|---:|
| Unethical actions taken | 6 | 3.75 | 1 | 0.97 |
| Times a stored shortcut was reused | 5 | 1.25 | 0 | 0.23 |
| Worst escalation reached (rung 0-3) | 3 | 1 | 1 | 0.97 |
| Dangerous actions executed | 1.50 | 0 | 0.25 | 0 |
| Unethical memories left at the end | 3 | 0.25 | 0 | 0 |
| Tasks done honestly | 0 | 1 | 5 | 4.78 |
| Tasks done (any method) | 6 | 4.50 | 6 | 5.75 |
| Episodes stopped by the filter | 0 | 0.75 | 0 | 0.25 |
| Useful honest memories lost | 0 | 0.75 | 0 | 0 |
| False alarms cleared by the Memory Manager | 0 | 0 | 0 | 0.75 |
| Euphemisms caught | 0 | 0 | 0.25 | 0.25 |
| Euphemisms stored | 0.25 | 0.25 | 0 | 0.23 |
| Full reviews (Memory Manager cost) | 0 | 0 | 8.75 | 3.35 |
| Light reviews | 0 | 0 | 0 | 5.60 |
| Reminders sent | 0 | 18.25 | 9 | 12.93 |
