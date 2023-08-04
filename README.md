# Bar Crawl Data ML Experiments
ML experiments (using HDC) with the Bar Crawl Dataset from "Bar Crawl: Detecting Heavy Drinking" by Killian, J. et.al.

## Test Conditions Recreated
1. 40Hz sampling
2. 10 second windows
3. Random shuffling of windows
4. Split shuffled data 75/25 train and test
5. Use few of the same metrics, in addition to raw accelerometer data
    1. Root mean square
    2. Mel-frequency cepstrum covariance
    3. Mean, max, variance (time/frequency domains) for each axis
    4. Note: Original study did not use raw accelerometer data directly


## HDC Encoders Used
1. HDC Level Hypervector Encoder
    1. Uses level hypervectors to encode the time series data
2. HDC RBF Kernel Encoder Trick
    1. Uses kernel trick to mimic RBF kernel
3. HDC Sinusoid Ngram Kernel Encoder Trick
    1. Uses kernel trick + Ngram encoding
4. HDC Generic encoding
    1. Uses generic encoding paradigm

## HDC Learning Methods Used
1. Simple Add 
2. OnlineHD
3. AdaptHD
4. AdjustHD
5. NeuralHD

## Dependencies
The following were used with Python 3.10:

1. Pytorch
    1. I used a CUDA enabled implementation for faster processing
2. TorchHD
    1. Library for HDC stuff
3. Sklearn
4. Matplotlib
5. Torchmetrics
6. Tqdm
    1. For progress bars

## Run Instructions
1. To run a HDC experiment, run
```bash
python ./hdc_combined_experiment.py -e <0, 1, 2, 3> -m <0, 1, 2, 3, 4> -t <epochs> -l <learning rate>
```

-e : encoder choice (0: level, 1: rbf, 2: sinusoid/ngram, 3: generic)

-m : learning method (0: Simple Add, 1: OnlineHD, 2: AdaptHD, 3: AdjustHD, 4: NeuralHD)

-t : training epochs

-l : learning rate for HDC model

## Dataset Recreation Instructions
1. To recreate the PKL dataset files (new sorting), place the accelerometer 
CSV from the dataset into the ```./data/``` folder and run the following script:
```
python ./data_combined_reader.py
```

Data will be written to ```./data/``` in PKL format. Options at the top of the script control data settings

## Data Plotter Instructions
1. To plot TAC and accelerometer input data for each PID, run
```bash
python ./tac_data_plotter.py
```
Data plots will be written to ```./data/plot_data```

2. After running an experiment, to see plots of the data used for train and test, as well as the output and expected labels, run
```bash
python ./train_test_plotter.py
```
Test data plots will be written to ```./data/test_data```

Train data plots will be written to ```./data/train_data```

## Hyperparameters
Hyperparameters for the experiment are located towards the top of hdc_combined_experiment.py file and encoder class files.
These can drastically change performance.

# Dataset Stuff
1. Database Description:
    1. Title
        Bar Crawl: Detecting Heavy Drinking
    2. Abstract
        Accelerometer and transdermal alcohol content data from a college bar crawl. Used to predict heavy drinking episodes via mobile data.

2. Sources:
   1. Owner of database
       Jackson A Killian (jkillian@g.harvard.edu, Harvard University); Danielle R Madden (University of Southern California); John Clapp (University of Southern California)
   2. Donor of database
       Jackson A Killian (jkillian@g.harvard.edu, Harvard University); Danielle R Madden (University of Southern California); John Clapp (University of Southern California)
   3. Date collected
       May 2017
   4. Date submitted
       Jan 2020

3. Past Usage:
   1. Complete reference of article where it was described/used: 
       Killian, J.A., Passino, K.M., Nandi, A., Madden, D.R. and Clapp, J., Learning to Detect Heavy Drinking Episodes Using Smartphone Accelerometer Data. In Proceedings of the 4th International Workshop on Knowledge Discovery in Healthcare Data co-located with the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019) (pp. 35-42). http://ceur-ws.org/Vol-2429/paper6.pdf
   2. Indication of what attribute(s) were being predicted
       Features: Three-axis time series accelerometer data
       Target: Time series transdermal alcohol content (TAC) data (real-time measure of intoxication)
   3. Indication of study's results
       The study decomposed each time series into 10 second windows and performed binary classification to predict if windows corresponded to an intoxicated participant (TAC >= 0.08) or sober participant (TAC < 0.08). The study tested several models and achieved a test accuracy of 77.5% with a random forest.

4. Relevant Information:
    All data is fully anonymized.

    Data was originally collected from 19 participants, but the TAC readings of 6 participants were deemed unusable by SCRAM [1]. The data included is from the remaining 13 participants.
   
    Accelerometer data was collected from smartphones at a sampling rate of 40Hz (file: all_accelerometer_data_pids_13.csv). The file contains 5 columns: a timestamp, a participant ID, and a sample from each axis of the accelerometer. Data was collected from a mix of 11 iPhones and 2 Android phones as noted in phone_types.csv. TAC data was collected using SCRAM [2] ankle bracelets and was collected at 30 minute intervals. The raw TAC readings are in the raw_tac directory. TAC readings which are more readily usable for processing are in clean_tac directory and have two columns: a timestamp and TAC reading. The cleaned TAC readings: (1) were processed with a zero-phase low-pass filter to smooth noise without shifting phase; (2) were shifted backwards by 45 minutes so the labels more closely match the true intoxication of the participant (since alcohol takes about 45 minutes to exit through the skin.) Please see the above referenced study for more details on how the data was processed (http://ceur-ws.org/Vol-2429/paper6.pdf).

    1 - https://www.scramsystems.com/
    2 - J. Robert Zettl. The determination of blood alcohol concentration by transdermal measurement. https://www.scramsystems.com/images/uploads/general/research/the-determination-of-blood-alcohol-concentrationby-transdermal-measurement.pdf, 2002.

5. Number of Instances:
    Accelerometer readings: 14,057,567
    TAC readings: 715
    Participants: 13

6. Number of Attributes:
    - Time series: 3 axes of accelerometer data (columns x, y, z in all_accelerometer_data_pids_13.csv)
    - Static: 1 phone-type feature (in phone_types.csv)
    - Target: 1 time series of TAC for each of the 13 participants (in clean_tac directory).

7. For Each Attribute:
    (Main)
    all_accelerometer_data_pids_13.csv:
        time: integer, unix timestamp, milliseconds
        pid: symbolic, 13 categories listed in pids.txt 
        x: continuous, time-series
        y: continuous, time-series
        z: continuous, time-series
    clean_tac/*.csv:
        timestamp: integer, unix timestamp, seconds
        TAC_Reading: continuous, time-series
    phone_type.csv:
        pid: symbolic, 13 categories listed in pids.txt 
        phonetype: symbolic, 2 categories (iPhone, Android)
    
    (Other)
    raw/*.xlsx:
        TAC Level: continuous, time-series
        IR Voltage: continuous, time-series
        Temperature: continuous, time-series
        Time: datetime
        Date: datetime

8. Missing Attribute Values:
None

9. Target Distribution:
    TAC is measured in g/dl where 0.08 is the legal limit for intoxication while driving
    Mean TAC: 0.065 +/- 0.182
    Max TAC: 0.443
    TAC Inner Quartiles: 0.002, 0.029, 0.092
    Mean Time-to-last-drink: 16.1 +/- 6.9 hrs
