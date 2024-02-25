# Amazon SageMaker MLOps pipeline with PrestoDB

Amazon SageMaker can be used to create end-to-end MLOps pipelines that include steps for data preparation, model training, tuning, evaluation and batch transform. In this repo we show how you can read raw data available in [`PrestoDB`](https://prestodb.io/) via SageMaker Processing Jobs, then train a binary classification model using SageMaker Training Jobs, tune the model using SageMaker Automatic Model Tuning and then run a batch transform for inference. All these steps are tied together via two SageMaker Pipelines: a training pipeline and a batch inference pipeline. Finally, we also demonstrate deploying the trained model on a SageMaker Endpoint for real-time inference.

## Getting started

To run this code follow the prerequisites below and then run the notebooks in the order specified.

### Prerequisites

The following prerequisites need to be in place before running this code.

#### PrestoDB

We will use the built-in datasets available in PrestoDB for this repo. Following the instructions below to setup PrestoDB on an Amazon EC2 instance in your account. ***If you already have access to a PrestoDB instance then you can skip this section but keep its connection details handy (see the `presto` section in the [`config`](./config.yml) file)***.

1. Create a security group to limit access to Presto. Create a security group called **MyPrestoSG** with two inbound rules to only allow access to Presto.
    - Create the first rule to allow inbound traffic on port 8080 to Anywhere-IPv4
    - Create the second rule rule to allow allow inbound traffic on port 22 to your IP only.
    - You should allow all outbound traffic

1. Spin-up an EC2 instance with the following settings. This instance is used to run PrestoDB.
    - AMI: Amazon Linux 2 AMI (HVM)
    - SSD Volume Type – `ami-0b1e534a4ff9019e0` (64-bit x86) / `ami-0a5c7dec456e07a8d` (64-bit Arm)
    - Instance type: `t3a.medium`
    - Subnet: Pick a public one and assign a public IP
    - IAM role: None
    - EBS: 8 GB gp2
    - Security group: MyPrestoSG

1. Install the JVM and Presto binaries.

    - Once the instance state changes to “running” and status checks are passed. Try to `ssh` into your EC2 instance with:

        ```{.bash}
        ssh ec2-user@{public-ip} -i {location}
        ```

    - If everything goes well, you will see the shell of your EC2 instance.

    - Install Presto 330 on the EC2 instance. Presto 330 requires the long-term support version Java 11. So let’s install it. First elevate yourself to root

        ```{.bash}
        sudo su
        ```

        Then update yum and install Amazon Corretto 11.

        ```{.bash}
        yum update -y
        yum install java-11-amazon-corretto.x86_64
        java --version
        ```

    - Now download the PrestoDB release binaries into the EC2 instance. You can download the Presto release binaries from the Maven Central Repository with `wget`. Then extract the archive to a directory named presto-server-330.

        ```{.bash}
        wget https://repo.maven.apache.org/maven2/io/prestosql/presto-server/330/presto-server-330.tar.gz 

        tar xvzf presto-server-330.tar.gz 

        ls -ltr presto-server-330
        ```

1. Configure Presto and add a data source. Before we start the Presto daemon, we must first provide a set of configuration files in `presto-server-330/etc` and add a data source. Go into presto-server-330 and create the `etc` directory

    ```{.bash}
    cd presto-server-330
    mkdir etc
    ```

    - Then create the three files using vim or your favorite text editor.
        - Presto logging configuration file `etc/config.properties`

            ```{.bash}
            coordinator=true
            node-scheduler.include-coordinator=true
            http-server.http.port=8080
            query.max-memory=5GB
            query.max-memory-per-node=1GB
            query.max-total-memory-per-node=2GB
            discovery-server.enabled=true
            discovery.uri=http://localhost:8080 
            ```

        - Presto node configuration: `etc/node.properties`

            ```{.bash}
            node.environment=demo 
            ```

        - JVM configuration: `etc/jvm.config`
        
            ```{.bash}
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
 
        - Catalog properties file for the TPC-H connector: `etc/catalog/tpch.properties` 

            ```{.bash}
            connector.name=tpch
            ```

1. Run the PrestoDB daemon. Use the bin/launcher script to start Presto as a foreground process. The script is in the folder **presto-server-330**

    ```{.bash}
    bin/launcher run
    ```

    If you’ve set everything up right, Presto will begin printing logs to stdout and stderr. After awhile you should see this line

    ```{.bash}
    INFO        main io.prestosql.server.PrestoServer ======== SERVER STARTED  
    ```

1. You have a running instance of PrestoDB! Since you launched PrestoDB on a public subnet and enabled 8080 inbound traffic. You can even access the UI at `http://{ec2-public-ip}:8080`.

#### IAM Role

The SageMaker execution role used to run this solution should have permissions to launch, list and describes various SageMaker services and artifacts. ***Until a AWS CloudFormation template is provided which creates the role with the requisite IAM permissions, use a SageMaker execution role that `AmazonSageMakerFullAccess` AWS managed policy for your execution role.

#### AWS Secrets Manager

Setup a secret in Secrets Manager for the PrestoDB username and password. Call the secret `prestodb-credentials` and add a `username` field to to it and a `password` field to it.

### Steps to run

1. Clone the [code repo](https://github.com/aws-samples/mlops-pipeline-prestodb.git) on SageMaker Studio.

1. Edit the [`config`](./config.yml) as per PrestoDB connection, IAM role and other pipeline details such as instance types for various pipeline steps etc.

    - [**Mandatory**] Edit the parameter values in the `presto` section.
    - [**Mandatory**]Edit the parameter values in the `aws` section.
    - [Optional] Edit the parameter values in the rest of the sections as appropriate.

1. Edit the [query.py](./code/query.py) file to replace the `TRAINING_DATA_QUERY` and `BATCH_INFERENCE_QUERY` values to your specific PrestoDB query. The queries in this repo are examples that use one of the built-in datasets available in PrestoDB.

1. Run the [`0_model_training_pipeline`](./0_model_training_pipeline.ipynb) notebook to train and tune the ML model and register it with the SageMaker model registry. All the steps in this notebook are executed as part of a training pipeline.
    - This notebook also contains an automatic model approval step that changes the state of the model registered with the model registry from `PendingForApproval` to `Approved` state. This step can be removed for prod accounts where manual or some criteria based approval would be required.

1. Run the [`1_batch_transform_pipeline.ipynb`](./1_batch_transform_pipeline.ipynb) notebook to launch the batch inference pipeline that reads data from PrestoDB and runs batch inference on it using the most recent `Approved` ML model.

1. Run the [`2_realtime_inference`](./2_realtime_inference.ipynb) notebook to deploy the model as a SageMaker endpoint for real-time inference.

## Contributing

Please read our [contributing guidelines](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/master/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
