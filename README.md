# prestodb-ml-pipeline
---

## Getting started


### Steps to configure Presto on your EC2 instance:

1. First, we will create a security group to limit access to Presto. Create a security group called **MyPrestoSG** with two inbound rules to make Presto only accessible. 
    * Create the first rule to allow inbound traffic on port 8080 to Anywhere-IPv4
    * Create the second rule rule to allow allow inbound traffic on port 22 to your IP only. 
    * You should allow all outbound traffic

2. Second, we will spin-up an EC2 instance to run Presto.
Launch 1 EC2 instance with the following settings

    * AMI: Amazon Linux 2 AMI (HVM), 
    * SSD Volume Type – ami-0b1e534a4ff9019e0 (64-bit x86) / ami-0a5c7dec456e07a8d (64-bit Arm)
    * Instance type: t3a.medium
    * Subnet: Pick a public one and assign a public IP
    * IAM role: None
    * EBS: 8 GB gp2
    * Security group: MyPrestoSG

3. Third, we will install the JVM and Presto binaries.

    * Once the instance state changes to “running” and status checks are passed. Try to ssh into your EC2 instance with
        ```
        ssh ec2-user@{public-ip} -i {location}
        ```
        If everything goes well, you will see the shell of your EC2 instance.
    * Now we will install Presto 330 on the EC2 instance. Presto 330 requires the long-term support version Java 11. So let’s install it. First elevate yourself to root
    
        ```
        sudo su
        ```

        Then update yum and install Amazon Corretto 11 
        ```
        yum update -y
        yum install java-11-amazon-corretto.x86_64
        java --version
        ```
    * Now we will download the Presto release binaries into the EC2 instance. You can download the Presto release binaries from the Maven Central Repository with wget. Then extract the archive to a directory named presto-server-330
        ```
        wget https://repo.maven.apache.org/maven2/io/prestosql/presto-server/330/presto-server-330.tar.gz 

        tar xvzf presto-server-330.tar.gz 

        ls -ltr presto-server-330
        ```
4. Fourth, we will configure Presto and add a data source. Before we start the Presto daemon, we must first provide a set of configuration files in presto-server-330/etc and add a data source. Go into presto-server-330 and create the etc directory
        ```
        cd presto-server-330
        mkdir etc
        ```
    * Then create the three files using vim or your favourite text editor.
        * Presto logging configuration file 
        **etc/config.properties** 
        
            ```
            coordinator=true
            node-scheduler.include-coordinator=true
            http-server.http.port=8080
            query.max-memory=5GB
            query.max-memory-per-node=1GB
            query.max-total-memory-per-node=2GB
            discovery-server.enabled=true
            discovery.uri=http://localhost:8080 
            ```
        * Presto node configuration
        **etc/node.properties** 
        
            ```
            node.environment=demo 
            ```
        * JVM configuration 
        **etc/jvm.config** 
        
            ```
            -server
            -Xmx4G
            -XX:+UseG1GC
            -XX:G1HeapRegionSize=32M
            -XX:+UseGCOverheadLimit
            -XX:+ExplicitGCInvokesConcurrent
            -XX:+HeapDumpOnOutOfMemoryError
            -XX:+ExitOnOutOfMemoryError
            -Djdk.nio.maxCachedBufferSize=2000000
            -Djdk.attach.allowAttachSelf=true 
            ```
        * Catalog properties file for the TPC-H connector
        **etc/catalog/tpch.properties** 
        
            ```
            connector.name=tpch
            ```

5. Finally, we will run the Presto daemon. Use the bin/launcher script to start Presto as a foreground process. The script is in the folder **presto-server-330**

    ```
    bin/launcher run
    ```
    If you’ve set everything up right, Presto will begin printing logs to stdout and stderr. After awhile you should see this line

    ```
    INFO        main io.prestosql.server.PrestoServer ======== SERVER STARTED  
    ```

Congratulations you have a running instance of Presto! Since you launched Presto on a public subnet and enabled 8080 inbound traffic. You can even access the UI at http://{ec2-public-ip}:8080


### SageMaker Pipeline Notebooks:

This repository contains three notebooks that demo the capabilities of SageMaker Pipelines. SageMaker Pipelines provides a customizable toolkit for every stage of the machine learning workflow. It offers deep customization and tuning options to suit diverse organizational needs. Users can tailor SageMaker Pipelines to their specific use cases or create reusable generic machine learning pipelines across multiple applications.

These notebooks demonstrate how SageMaker Pipelines can be used to create a generic binary classification machine learning pipeline using Random Forest from order data retrieved by Presto. 

* [mlops_pipeline_train_pipeline_0](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/mlops_pipeline_train_pipeline_0.ipynb): The purpose of this notebook is to demonstrate how SageMaker Pipelines can be used to create a generic Random Forest training pipeline that preprocesses, trains, tunes, evaluates and registers new machine learning models with the SageMaker model registry, that is reusable across teams, customers and use cases. All scripts to preprocess the data and evaluate the trained model have been prepared in advance and are available in the code folder
* [mlops_pipeline_batch_transform_1](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/mlops_pipeline_batch_transform_1.ipynb): This notebook shows how to create a pipeline that reads the latest model registered in a model registry and perform batch transformation on data.
* [mlops_pipeline_model_deploy_2](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/mlops_pipeline__model_deploy_2.ipynb): This notebook shows how to create a pipeline that deploys the latest model registered in a model registry as a real time endpoint for inference

### Running Notebooks

To get started, navigate to the notebook and proceed to configure your parameters in the `config.yml` file, where you can tailor the settings for your pipeline. Ensure to update the feature and target parameters according to your desired model predictions. If you're using a Presto instance, it's essential to ensure accurate configuration of the host, port, and user parameters.

Additionally, the `config.yml` file provides flexibility to modify general pipeline and training parameters as needed. Once configurations are set, we can proceed with the first notebook

#### [Training Pipeline](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/mlops_pipeline_train_pipeline_0.ipynb)

This notebook demonstrates the capabilities of SageMaker Pipelines in crafting reusable machine learning pipelines. The pipeline orchestrates various steps, including preprocessing, training, evaluation, and model registration with the SageMaker Model Registry.

In this notebook, we follow these steps:
- Load the `config.yml` file containing crucial information used across the pipeline.
- Connect to Presto and query data for preprocessing, subsequently transferring it to an Amazon S3 bucket where it's partitioned into train, test, and validation datasets.
- Utilize the train and validation outputs from the preprocessing stage to train a model.
- Evaluate the model's performance, ensuring its accuracy exceeds a configurable threshold before registering it with the model registry.

#### [Batch Transform Pipeline](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/mlops_pipeline_batch_transform_1.ipynb)

This notebook illustrates the creation of a pipeline designed specifically for batch transformation of data, assuming a previously trained model is registered in a model registry.

Here, we assume that the training pipeline has successfully trained a model, registered it in the Model Registry, and obtained approval for inference. The workflow involves:
- Employing a pre-trained model, creating, and registering it in a new Model Registry.
- Executing batch inference using the approved model on a dataset stored in an S3 bucket, providing it as input to the pipeline.
- Processing and sending batch data from PrestoDB to Amazon S3.
- Utilizing the batch data for the batch transform and inference step, recording start and end times, and sending the output to an S3 path.

#### [Real-time Endpoint Pipeline](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/mlops_pipeline__model_deploy_2.ipynb)

This constitutes the third part of our solution, where we:
- Utilize the latest approved model to create a real-time endpoint.
- Run inferences to test the real-time deployed endpoint.

## Contributing
Please read our [contributing guidelines](https://github.com/aws/amazon-sagemaker-examples/blob/master/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

