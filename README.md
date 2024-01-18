# ECE 284: Assignment 1 and 2

## Contents
* [Deadlines](#deadlines)
* [Overview](#overview)
* [Setting up](#setting-up)
* [Code development and testing](#code-development-and-testing)
* [Submission guidelines](#submission-guidelines)

## Deadlines
- Assignment 1: Wednesday, Jan 24 2024 (by 11:59pm PT)
- Assignment 2: Wednesday, Jan 31 2024 (by 11:59pm PT)

## Overview

Assignment 1 and 2 have the same codebase. The program (`seedTable`) reads in an input sequence in the FASTA format using the `kseq` library (http://lh3lh3.users.sourceforge.net/kseq.shtml) that stores the sequence as a character array (i.e. one byte per character). The program converts this sequence into an array of `uint32_t` with two-bit compression, i.e. representing each character of the DNA sequence using only two bits (A: 2'b00, C: 2'b01, G: 2'b10, T: 2'b11). Next, it transfers the compressed sequence to a GPU device, on which a seed table, consisting of (i) kmer offset table and (ii) kmer position table is contructed in four steps: (i) finding all kmers in the sequence and storing a combined (kmer, kmerPosition) value using a 64-bit uinsigned integer array (ii) sorting this unsigned array using a parallel sort algorithm implemented in the thrust library (https://thrust.github.io/), (iii) iterating over this array to fill out the kmer offset table and (iv) masking the upper 32 bits of the concatenated array to form the kmer position table.

In Assignment 1, students are required to parallelize the two-bit compression step using the TBB library.

In Assignment 2, students are required to parallelize the different stages of the seed table construction using CUDA.

## Setting up

In ECE 284, we will be using UC San Diego's Data Science/Machine Learning Platform ([DSMLP](https://blink.ucsd.edu/faculty/instruction/tech-guide/dsmlp/index.html)), which provides students with access to research-class CPU and GPU resources for coursework and projects. ECE 284 Winter 2023 students would already have an account on DSMLP cluster which is linked to the Active Directory (AD) accounts. More detailed information about the DSMLP platform can be found at: 
* [https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0030470](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0030470)
* [https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269)
* [https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273)
  
Please go through the above pages once so that you get a better idea of the compute infrastructure, though the instructions below should be sufficent for solving the assignments.

Broadly, students will submit jobs on the DSMLP cluster that will be executed in the form of Docker “containers” which are essentially lightweight virtual machines, each assigned dedicated CPU, RAM, and GPU hardware, and each well isolated from other users’ processes.

To get set up with Assignment 1 and 2, please follow the steps below:

1. Open and accept the following GitHub Classroom invitation link for assignments 1 and 2 through your GitHub account: [https://classroom.github.com/a/QobW_O9J](https://classroom.github.com/a/QobW_O9J). A new repository for this will be created specifically for your account (e.g. https://github.com/ECE284-WI24/assgn1-2-yatisht) and an email will be sent to you via GitHub with the details.

2. SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. I recommend using PUTTY SSH client (putty.org) or Windows Subsystem for Linux (WSL) for Windows (https://docs.microsoft.com/en-us/windows/wsl/install-manual). MacOS and Linux users can SSH into the server using the following command (replace `yturakhia` with your username)

```
ssh yturakhia@dsmlp-login.ucsd.edu
```

3. Next, clone the assignment repository in your HOME directory using the following example command (replace repository name `assgn1-2-yatisht` with the correct name based on step 1):
```
cd ~
git clone https://github.com/ECE284-WI24/assgn1-2-yatisht
```

4. Download a copy of the TBB version 2019_U9 into your HOME directory:

```
wget https://github.com/oneapi-src/oneTBB/archive/2019_U9.tar.gz
tar -xvzf 2019_U9.tar.gz
```

5. Review the source code (in the `src/` directory). In particular, search `TASK` (e.g. in `main.cpp` and `twoBitCompressor.cpp` for tasks related to Assignment ) and `HINT` in these files. Also review the `run-commands.sh` script. This script contains the commands that will be executed via the Docker container on the GPU instance. You may need to modify the commands of this script depending on your experiment. Finally, make sure to also review the input test data files in the `data` directory.
```
cd assgn1-2-yatisht
```

## Code development and testing

Once your environment is set up on the DSMLP server, you can begin code development and testing using either VS code (that many of you must be familiar with) or if you prefer, using the shell terminal itself (with text editors, such as Vim or Emacs). If you prefer the latter, you can skip the step 1 below.

1. Launch a VS code server from the DSMLP login server using the following command:
   ```
   /opt/launch-sh/bin/launch-codeserver -i ucsdets/datascience-notebook:2022.2-stable
   ```
   If successful, the log of the command will include a message such as:
   ```
   You may access your Code-Server (VS Code) at: http://dsmlp-login.ucsd.edu:14672 using password XXXXXX
   ```
   If the launch command is *unsuccessful*, make sure that there are no aleady running pods:
   ```
   # View running pods
   kubectl get pods
   # Delete all pods
   kubectl delete pod --all
   ```
   As conveyed in the message of the successful launch command, you can access the VS code server by going to the URL above (http://dsmlp-login.ucsd.edu:14672 in the above example) and entering the password displayed. Note that you may need to use UCSD's VPN service (https://blink.ucsd.edu/technology/network/connections/off-campus/VPN/) if you are performing this step from outside the campus network. Once you gain access to the VS code server from your browser, you can view the directories and files in your DSMLP filesystem and develop code. You can also open a terminal (https://code.visualstudio.com/docs/editor/integrated-terminal) from the VS code interface and run commands on the login server.

2. As mentioned before, we will be using a Docker container, namely `yatisht/ece284-wi24:latest`, for submitting a job on the cluster containing the right virtual environment to build and test the code. This container already contains the correct Cmake version, CUDA and Boost libraries preinstalled within Ubuntu-20.04 OS. Note that these Docker containers use the same filesystem as the DSMLP login server, and hence the files written to or modified by the conainer is visiable to the login server and vice versa. To submit a job that executes `run-commands.sh` script located inside the `assgn1-2-yatisht` direcotry on a VM instance with 8 CPU cores, 8 GB RAM and 1 Rtx 2080 Ti GPU device (1 GPU is the maxmimum allowed request on the DSMLP platform), the following command can be executed from the VS Code or DSMLP Shell Terminal (replace the username and directory names below appropriately):

```
ssh yturakhia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -v 2080ti -c 8 -g 1 -m 8 -i yatisht/ece284-wi24:latest -f ./assgn1-2-yatisht/run-commands.sh
```
Note that the above command will require you to enter your AD account password again. This command should work and provide a sensible output for the assignment already provided. If you have reached this, you are in good shape to develop and test the code (make sure to modify `run-commands.sh` appropriately before testing). Happy code development!

## Submission guidelines

* Make sure to keep your code repository (e.g. https://github.com/ECE284-WI24/assgn1-2-yatisht) up-to-date.
* All new files (such as figures and reports) should be uploaded to the `submission-files` directory in the respository
* Once you are ready to submit, create a [new release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) for your submission with tag names shown below. Provide a good description for the changes you have made and any information that you would like to be conveyed during the grading.
  * Assignment 1: `v1.0`
  * Assignment 2: `v2.0`
* Submit the URL corresponding to the release to Canvas for your assignment submission (e.g. https://github.com/ECE284-WI24/assgn1-2-yatisht/releases/tag/v1.0; note that you are only required to submit a URL via Canvas). Only releases will be considered for grading and the date and time of the submitted release will be considered final for the late day policy. Be mindful of the deadlines.

