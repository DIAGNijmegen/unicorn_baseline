# Automated Local Testing  
We provide scripts to automate the local testing process using public few-shot data from Zenodo:

1. Clone this repository:
   ```bash
   git clone https://github.com/DIAGNijmegen/unicorn_baseline.git
   cd unicorn_baseline
   ```
2. Build the Docker container:
   ```
   ./do_build.sh
   ```
3. Perform test run(s):

- `run_task.sh`: Downloads and prepares data for a single task, then runs a test on one case. Note: Make sure to always take the **latest** version of the data on Zenodo.
   
   ```bash 
   ./run_task.sh "https://zenodo.org/records/15315589/files/Task01_classifying_he_prostate_biopsies_into_isup_scores.zip"
   ```
- `run_all_tasks.sh`: Runs the above process for all 20 UNICORN tasks sequentially.   
   ```bash
      ./run_all_tasks.sh  
   ```
- `do_test_run.sh`: Directly test your Docker container on a specific case folder. 
  
   ```bash
  ./do_test_run.sh path/to/case/folder [docker_image_tag]
  ```

4. Save the container for upload:
   ```bash
   ./do_save.sh
   ```