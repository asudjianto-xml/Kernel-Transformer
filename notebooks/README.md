# notebooks/

This directory will hold one notebook per chapter (`NN_topic.ipynb`).
Notebooks are added by chapter agents (CH01–CH26).

To run a notebook end-to-end:
    papermill notebooks/NN_topic.ipynb /tmp/output.ipynb

CI validates that all notebooks run cleanly on every PR.
