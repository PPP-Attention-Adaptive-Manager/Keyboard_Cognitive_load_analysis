README: Keyboard Cognitive Load Analysis Pipeline

This document outlines the end-to-end workflow for the Keyboard Analysis project, covering data ingestion, feature engineering, machine learning training, and real-time testing.
Step 1: Data Merging (data_merging.py)

Goal: Consolidate fragmented raw logs into a single master dataset.

    Input: Multiple individual CSV or TXT files from different participants or sessions.

    Process: The script standardizes column headers (e.g., ensuring PRESS_TIME and RELEASE_TIME are consistent) and merges them into a single dataframe.

    Output: merged_first_sections.csv.

Step 2: Data Understanding (data_understanding.py)

Goal: Verify data integrity and encoding before processing.

    Action: Scans the LETTER and KEYCODE columns to identify how deletions are recorded.

    Key Check: It confirms if backspaces are labeled as 'Backspace', 'BKSP', or ASCII code 8. If these are not found, the BACKSPACE_RATIO feature will remain at zero during training.

Step 3: Data Processing (data_processing.py)

Goal: Transform raw "event-level" data (16M+ rows) into "session-level" features.

    Process: Groups the raw keystrokes by PARTICIPANT_ID and TEST_SECTION_ID.

    Feature Engineering: Calculates the following metrics:

        IKI_CV: The Coefficient of Variation of Inter-Keystroke Intervals (measures rhythm consistency).

        PAUSE_RATIO: The total time spent in pauses divided by the session duration.

        PAUSE_PER_MIN: The frequency of hesitations per minute.

    Output: session_cogload_metrics.csv.

Step 4: Model Training (model_training.py)

Goal: Train a Random Forest Regressor to predict a mathematically derived cognitive load proxy.

    The Proxy Formula: The target variable is a complex non-linear formula:
    0.3⋅ln(1+Hold⋅Pause)+0.2⋅Max_Hold​+0.2⋅Pause_Per_Min1.5…

    Process:

        Scales the independent features using StandardScaler.

        Trains a Random Forest Regressor (200 trees) to learn the underlying patterns.

        Evaluates performance using R-squared (R2) and Mean Squared Error (MSE).

    Output: * rf_cogload_model_complex.pkl (The trained model).

        feature_scaler_complex.pkl (The scaling parameters).

Step 5: Testing Mode (testing_model.py)

Goal: Deploy the model in a real-time GUI for live monitoring.

    Sliding Window: The script only analyzes keystrokes occurring within the last 60 seconds.

    The Frontier Logic: To differentiate between "Struggling" and "Thinking," the script applies a 5-second frontier.

        Pauses 1s - 5s: Classified as Hesitation (Increases load).

        Pauses > 5s: Classified as Reflection (Stabilizes or lowers load).

    Hybrid Averaging: The final output is calculated as the average of the Machine Learning prediction and the manual Proxy Math:
    Final Load=2ML Prediction+Proxy Formula​

Status Indicators and Thresholds

    0.0 - 3.5 (Low): Active flow. Rhythmic typing with minimal corrections.

    3.5 - 7.0 (Moderate): Increasing load. Choppy typing or frequent hesitations.

    7.0 - 10.0 (High): Significant cognitive stress. High error rates and frequent stalling.

    Idle/Reflection: Triggered when the "Frontier" is crossed, indicating the user has stopped to reflect or has left the station.