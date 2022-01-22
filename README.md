# Covid-19_Detector
Building and deploying TensorFlow Deep CNN models to a Web-App for Covid Detection using Chest X-ray &amp; CT-scan Images

## Commands to run Web-App Locally

### Initial setup

Activate your environment in terminal then run following command, If python is locally installed in windows then use command propmt (with Admin Access) or windows powershell 

This command will install all necessary packages to run the Web-App

```pip install -r requirements.txt```

### To run the Application on local server

* Navigate to the Covid-19_Detector Folder then run the following command,

* To run the application you can either use the flask command or python’s ```-m``` switch with Flask. Before you can do that you need to tell your terminal the application to work with by exporting the ```FLASK_APP``` environment variable:

  ```$ export FLASK_APP=app.py```

  ```$ flask run```

  Output : ```* Running on http://127.0.0.1:5000/```
 
* If you are on Windows, the environment variable syntax depends on command line interpreter. On Command Prompt:

  ```C:\path\to\app>set FLASK_APP=app.py```

* And on PowerShell:

  ```PS C:\path\to\app> $env:FLASK_APP = "app.py"```

* Alternatively you can use ```python -m flask```:

  ```$ export FLASK_APP=app.py```

  ```$ python -m flask run```

  Output : ```* Running on http://127.0.0.1:5000/```
 
 ### To run the Application on AWS-EC2 Server
 
 #### Here you can use ```app_aws.py``` to utilize the AWS runtime with limited resources, this file is modified to use only limited models to predict the results, to save memory
 
 * Create Your AWS-EC2 instance (Ubuntu Only):
 
    <a href="https://docs.aws.amazon.com/efs/latest/ug/gs-step-one-create-ec2-resources.html" target="_blank">Create AWS-EC2 Instance</a>
 
 * Now ```SSH``` into your instance using any method, I prefer to use ```WinSCP``` and ```PuTTY``` software using windows but you can follow any oher methods too, If you are using ```PuTTY``` make sure you convert ```.pem``` access key to ```.ppk``` using ```PuTTYgen``` software
 
 * Follow Initial Setup Commands mentioned above to install packages, make sure you have python version ```>3.8``` on instance
 
 * Now run the following command on terminal of AWS-EC2 instance, to create a session which will run application even if you are not online on your system,
 
    ```$ tmux new -s <session_name>```
 
 * Run Commands to run your Application, make sure you notedown your port address which in my case was ```:8080```, this info will be available at very last line of your ```app_aws.py```
 
 * Now you can close your ```SSH``` connection,
 
 * Go to your AWS-EC2 instance panel, click on the link of your ```Public IPv4 DNS``` (refer Image shown below),
 
   ![image](https://user-images.githubusercontent.com/55990500/150626923-cf9372a1-cb73-42a3-8825-d66481321eb3.png)
 
 * Add port ```:8080``` add the end of your ```Public IPv4 DNS``` link as shown below,
 
    ```http://ec2-52-66-147-151.ap-south-1.compute.amazonaws.com:8080/```
 
 * To re-login into your ```tmux``` session use command,
 
    ```tmux a -t <session_name>```
 
 * To exit from ```tmux``` session use command,
 
    ```exit```
 
 * You can refer ```tmux``` documentation on,
 
   <a href="https://tmuxguide.readthedocs.io/en/latest/tmux/tmux.html" target="_blank">tmux_docs</a>

 

