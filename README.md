# omega

(Execute these commands in Git Bash:)Tiger/Lion Server Usr & Pass:
1- % ssh indigo.eecs.yorku.ca -l <EECS USERNAME>  or
% ssh red.eecs.yorku.ca -l <EECS USERNAME>
pwd: yousefRAYENmed1
 
 
https://wiki.eecs.yorku.ca/dept/tdb/login:sshsupport
2- ssh lion or ssh tiger
pwd: hajaya1*
 
3- Conda activate pyt_tns
4- cd omega/Omega-notebook
5- git pull https://github.com/SepidehNahali/omega.git
6- python launch.py
ــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
GitHub Setting:
UserName: SepidehNahali
Git token: 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
(you should create the token in git account then add it to Credential windows on your computer to be able to clone project from git repo to the server)
ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
(Execute these commands in Git Bash:)To Get the Latest Update Of the Cloned repo:
ــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
(Execute these commands in Git Bash:)For linon server: git pull https://github.com/SepidehNahali/omega.git
For colab: No need as we clone each time.
To transfer files from server to github:
1-  Cd to the directory of the file you want to copy it to the repository then:
2-  git add .
3-  git commit -m "Add existing file"
 
(by git branch I got that my branch name is :’ main’)
4-  git push origin main
 
 
All commands for copy to git repo:
git add .
git commit -m "Add existing file"
git push origin main
 
 
ـــــــــــــــــــــــــــــــــــــــــــــــــــ
(Execute these commands in Colab Notebook:)To show the tensorboard in google colab:
 
!git clone https://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@github.com/SepidehNahali/omega.git
%load_ext tensorboard
%tensorboard --logdir omega/Omega-notebooks/
!ls
 
 
 

