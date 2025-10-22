#!/bin/bash
# Submit all resume training jobs

echo 'Submitting resume training jobs...'

echo 'Submitting: DyGMamba/lastfm/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_dygmamba_lastfm_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: DyGMamba/mooc/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_dygmamba_mooc_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: DyGMamba/uci/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_dygmamba_uci_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: JODIE/Contacts/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_jodie_contacts_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: JODIE/SocialEvo/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_jodie_socialevo_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: JODIE/uci/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_jodie_uci_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: JODIE/wikipedia/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_jodie_wikipedia_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: TCL/Contacts/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_tcl_contacts_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: TCL/lastfm/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_tcl_lastfm_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: TCL/mooc/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_tcl_mooc_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: TCL/reddit/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_tcl_reddit_time2vec.sh
sleep 2  # Small delay between submissions

echo 'Submitting: TGN/Flights/time2vec'
qsub /home/s2516027/kan-mammotev2/resume_jobs/resume_tgn_flights_time2vec.sh
sleep 2  # Small delay between submissions

echo 'All jobs submitted!'
echo 'Check status with: qstat -u $USER'
