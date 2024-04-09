# How Twilio used Amazon SageMaker MLOps Pipelines with PrestoDB to enable frequent model re-training and optimized batch transform


*Amit Arora*, *Madhur Prashant*, *Antara Raisa*, *Johnny Chivers*

***This post is co-written with customer_names from Twilio.***

\[PLACEHOLDER (Twilio to add information here): Twilio is an American
cloud communications company based in San Francisco, California, which
provides programmable communication tools for making and receiving phone
calls, sending and receiving text messages, and performing other
communication functions using its web service APIs.\] Being one of the
largest AWS customers, Twilio engages with Data and AI/ML services to
run their daily workloads. This blog revolves around the steps AWS and
Twilio took to migrate Twilio’s existing Machine Learning Operations
(MLOps), implementation of training models and running batch inferences
to Amazon SageMaker.

Machine learning (ML) models do not operate in isolation. To deliver
value, they must integrate into existing production systems and
infrastructure, which necessitates considering the entire ML lifecycle
during design and development. With the right processes and tools, MLOps
enables organizations to reliably and efficiently adopt ML across their
teams for their specific use cases. [Amazon SageMaker
MLOps](https://aws.amazon.com/sagemaker/mlops/?sagemaker-data-wrangler-whats-new.sort-by=item.additionalFields.postDateTime&sagemaker-data-wrangler-whats-new.sort-order=desc)
is a suite of features that includes [Amazon SageMaker
Pipelines](https://aws.amazon.com/sagemaker/pipelines/), that allows for
straightforward creation and management of ML workflows, while also
offering storage and reuse capabilities for workflow steps and [Amazon
SageMaker Model
Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
that centralizes model tracking, simplifying model deployment.

This blog post focuses on enabling AWS customers to have flexibility for
using their data source of choice, and integrate it seamlessly with
[Amazon SageMaker Processing
Jobs](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker_processing/scikit_learn_data_processing_and_model_evaluation/scikit_learn_data_processing_and_model_evaluation.html),
where you can leverage a simplified, managed experience to run data pre-
or post-processing and model evaluation workloads on the Amazon
SageMaker platform.

[Twilio](https://pages.twilio.com/twilio-brand-sales-namer-1?utm_source=google&utm_medium=cpc&utm_term=twilio&utm_campaign=G_S_NAMER_Brand_Twilio_Tier1&cq_plac=&cq_net=g&cq_pos=&cq_med=&cq_plt=gp&gad_source=1&gclid=CjwKCAjwtqmwBhBVEiwAL-WAYd5PgxP-XSLDYBvu6y_j8KUydoj33QX3XWpUo4zEm2DLzgn_bfdogBoC9dIQAvD_BwE)
needed to implement an MLOPs pipeline and query data as a part of this
process from [PrestoDB](https://prestodb.io/). PrestoDB is an
open-source SQL query engine that is designed for fast analytic queries
against data of any size from multiple sources.

In this post, we show you a step-by-step implementation to achieve the
following:

-   How you can read data available in PrestoDB via a SageMaker
    Processing Job

-   Train a binary classification model using [SageMaker Training
    Jobs](https://sagemaker.readthedocs.io/en/v1.44.4/amazon_sagemaker_operators_for_kubernetes_jobs.html)
    and tune the model using [SageMaker Automatic Model
    Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)

-   Run a [Batch Transform
    pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
    for batch inference on data fetched from PrestoDB

-   Deploy the trained model as a [Real-Time SageMaker
    Endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)

## Use case overview

Burners are phone numbers which are available online to everyone and are
used to hide identities by creating fake accounts on customers’
apps/websites. Twilio built an MLOps pipeline to detect these anomalous
phone numbers with the help of a binary classification model using the
[scikit-learn](https://scikit-learn.org/stable/)
[`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
The training data they used for this pipeline is made available via
PrestoDB and is read into Pandas through the [PrestoDB Python
client](https://pypi.org/project/presto-python-client/).

The end goal was to convert all the existing steps into an
implementation of a training pipeline, a batch transform pipeline (by
connecting a SageMaker Processing Job with data queried from PrestoDB),
and deploying the trained model on a SageMaker Endpoint for real-time
inference.

Twilio was able to use this open-source solution to migrate their burner
model and MLOps to Amazon SageMaker. In this blog, we use the
[TPCH-Connector](https://prestodb.io/docs/current/connector/tpch.html)(this
allows users to test PrestoDB’s capabilities and query syntax without
needing to configure access to an external data source) as our data
source. All the code for this post is available in the
[GitHub](https://github.com/aws-samples/mlops-pipeline-prestodb?tab=readme-ov-file)
repo.

## Solution overview

This solution is divided into three main steps: training pipeline, batch
transform pipeline, and deploying the trained model as a real time
SageMaker Endpoint for inference as follows:

-   [Model Training
    Pipeline](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/0_model_training_pipeline.ipynb):
    In this step, we create a model training pipeline. We connect a
    SageMaker Processing Job to fetch data from a PrestoDB instance,
    train and tune the ML model and register it with the SageMaker Model
    Registry. All the steps in this notebook are executed as part of the
    training pipeline.
-   [Batch Transform
    Pipeline](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/1_batch_transform_pipeline.ipynb):
    In this step, we create a batch transform pipeline. Here, we execute
    the batch preprocess data step that reads data from the PrestoDB
    instance and runs batch inference on the registered ML model that we
    [`Approve`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html)
    as a part of this pipeline. This model is approved either
    programmatically or manually via the Model Registry.
-   [Real-time
    Inference](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/2_realtime_inference.ipynb):
    In this step, we deploy the latest approved model as a SageMaker
    Endpoint for [Real-Time
    inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html).

## Solution design

The solution design consists of the following parts: Setting up the data
preparation and training pipeline, preparing for the batch transform
pipeline, and deploying the approved model as a real time SageMaker
Endpoint for inference. All [pipeline
parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-parameters.html)
used by this solution exist in this single
[config.yml](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/config.yml)
file. This file includes the necessary AWS and PrestoDB credentials to
connect to the PrestoDB instance, information on the training
[hyperparameters](https://sagemaker.readthedocs.io/en/stable/api/utility/hyperparameters.html)
that are used and
[SQL](https://aws.amazon.com/what-is/sql/#:~:text=Structured%20query%20language%20(SQL)%20is,relationships%20between%20the%20data%20values.)
queries that are run at training and inference steps. This design makes
this solution highly customizable for industry specific use cases so
that it can be personalized and used with minimal-no code changes.

An example of how a query is configured within this file is given below.
This query is used at the data preprocessing step, to fetch data from
the PrestoDB instance on a multi classifier problem. Here, we predict
whether an `order` is a `high_value_order` or a `low_value_order` based
on the `orderpriority` as given from the `TPCH-data`. Customers and
users can change the query for their use case simply within the config
file and run the solution as is without making any internal code
changes.

``` sql
    SELECT
        o.orderkey,
        COUNT(l.linenumber) AS lineitem_count,
        SUM(l.quantity) AS total_quantity,
        AVG(l.discount) AS avg_discount,
        SUM(l.extendedprice) AS total_extended_price,
        SUM(l.tax) AS total_payable_tax,
        o.orderdate,
        o.orderpriority,
        CASE
            WHEN (o.orderpriority = '2-HIGH') THEN 1 
            ELSE 0
        END AS high_value_order
    FROM
        orders o
    JOIN
        lineitem l ON o.orderkey = l.orderkey
    GROUP BY
        o.orderkey,
        o.orderdate,
        o.orderpriority
    ORDER BY 
        RANDOM() 
    LIMIT 5000
```

The main components of this solution are as described in detail below:

### Part 1 - [Data Preparation and Training Pipeline Step](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/0_model_training_pipeline.ipynb):

1.  The training data is read from a PrestoDB instance, and any feature
    engineering needed is done as part of the SQL queries run in
    PrestoDB at retrieval time. The queries used to fetch data at the
    training and batch inference step are configured in the [config
    file](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/config.yml).
2.  We use the
    [FrameworkProcessor](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job-frameworks.html)
    with SageMaker Processing Jobs to read data from PrestoDB using the
    Python PrestoDB client.
3.  For the training and tuning step, we use the [SKLearn
    estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html)
    from the SageMaker SDK and the `RandomForestClassifier` from
    `scikit-learn` to train the ML model. The
    [HyperparameterTuner](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html)
    class is used for running automatic model tuning to determine the
    set of hyperparameters that provide the best performance for a given
    use case (for example, maximize the [AUC
    metric](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)).
4.  The [Model
    Evaluation](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.html)
    step is to check that the trained and tuned model has an accuracy
    level above a given threshold and only then [register the
    model](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
    within the Model Registry. If the model accuracy does not meet the
    given threshold then the pipeline fails and the model is not
    registered with the Model Registry.
5.  The model training pipeline is then run with the
    [`pipeline.start`](https://docs.aws.amazon.com/sagemaker/latest/dg/run-pipeline.html)
    which triggers and instantiates all steps above.

### Part 2 - [Batch Transform Step](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/1_batch_transform_pipeline.ipynb):

1.  The batch transform pipeline consists of two steps: a data
    preparation step that retrieves data from a PrestoDB instance (using
    a [batch data preprocess
    script](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/presto_preprocess_for_batch_inference.py))
    and stores the batch data in S3. After this, a batch transform step
    runs inference on this data stored in S3 and stores the output data
    in S3.
2.  We then utilize the
    [Transformer](https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html)
    instance to get inferences on the entire batch dataset queried from
    PrestoDB which is stored in S3.

### Part 3 - [Real Time SageMaker Endpoint support](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/2_realtime_inference.ipynb):

1.  The latest approved model is retrieved from the Model Registry using
    the
    [describe_model_package](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/describe_model_package.html)
    function from the SageMaker SDK.
2.  The latest approved model from the Model Registry is deployed as a
    real-time SageMaker Endpoint.
3.  The model is deployed on a `ml.c5.xlarge` instance with a minimum
    instance count of 1 and maximum instance count of 3 (configurable by
    the user) with the [automatic scaling
    policy](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
    `ENABLED`. This removes nnecessary instances so that you don’t pay
    for provisioned instances that you aren’t using.

## Prerequisites

To implement the solution provided in this post, you should have an [AWS
account](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fportal.aws.amazon.com%2Fbilling%2Fsignup%2Fresume&client_id=signup),
set up an [Amazon SageMaker
Domain](https://docs.aws.amazon.com/sagemaker/latest/dg/sm-domain.html)
to access [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/)
and familiarity with SageMaker, S3, and PrestoDB.

The following prerequisites need to be in place before running this
code.

#### PrestoDB

-   We use the built-in datasets available in PrestoDB via the
    `TPCH-connector` for this solution. Follow the instructions in the
    GitHub
    [README.md](https://github.com/aws-samples/mlops-pipeline-prestodb?tab=readme-ov-file#prestodb)
    to setup PrestoDB on an Amazon EC2 instance in your account. ***If
    you already have access to a PrestoDB instance then you can skip
    this section but keep its connection details handy (see the `presto`
    section in the [`config`](./config.yml) file)***. Once you have your
    PrestoDB credentials, fill out the `presto` section in the
    [`config`](./config.yml) as given below. Enter your host public IP,
    port, credentials, catalog and schema:

``` yaml
presto:
  host: <0.0.0.0>
  parameter: "0000"
  presto_credentials: <presto_credentials>
  catalog: <catalog>
  schema: <schema>
```

#### [Amazon VPC](https://aws.amazon.com/vpc/) Network Configurations

-   We also define the network configurations of the machine learning
    model and operations in the [`config`](./config.yml) file. In the
    `aws` section, specify the `enable_network_isolation` status,
    `security_group_ids`, and `subnets` based on your network isolation
    preferences. View more information on network configurations
    [here](https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html):

``` yaml
network_config:
    enable_network_isolation: false
    security_group_ids: 
    - <security_group_id>
    subnets:
    - <subnet-1>
    - <subnet-2>
    - <subnet-3>
```

#### IAM Role

Set up an execution role in [AWS Identity and Access Management
(IAM)](https://aws.amazon.com/iam/) with appropriate permissions to
allow SageMaker to access [AWS Secrets
Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html),
[Amazon
S3](https://aws.amazon.com/pm/serv-s3/?gclid=Cj0KCQjw2a6wBhCVARIsABPeH1sVCmK3CK8Vsv31A4fjV79s5YkxGqKoyDuv2rPuoBDfDqwh7ZiYaTQaAkeOEALw_wcB&trk=fecf68c9-3874-4ae2-a7ed-72b6d19c8034&sc_channel=ps&ef_id=Cj0KCQjw2a6wBhCVARIsABPeH1sVCmK3CK8Vsv31A4fjV79s5YkxGqKoyDuv2rPuoBDfDqwh7ZiYaTQaAkeOEALw_wcB:G:s&s_kwcid=AL!4422!3!536452728638!e!!g!!amazon%20s3!11204620052!112938567994)
and other services within your AWS account. ***Until a AWS
CloudFormation template is provided which creates the role with the
requisite IAM permissions, use a SageMaker execution role that
`AmazonSageMakerFullAccess` AWS managed policy for your execution
role.*** Follow the instructions
[here](https://github.com/aws-samples/amazon-sagemaker-w-snowflake-as-datasource/tree/main/iam)
to create permissions for your `iam` roles.

#### AWS Secrets Manager

Setup a secret in Secrets Manager for the PrestoDB username and
password. Call the secret `prestodb-credentials` and add a `username`
field to it and a `password` field to it. For instructions on creating
and managing secrets via `Secrets Manager`, view
[this](https://docs.aws.amazon.com/secretsmanager/latest/userguide/managing-secrets.html).

### Steps to run

1.  Clone the [code
    repo](https://github.com/aws-samples/mlops-pipeline-prestodb.git) on
    SageMaker Studio.

2.  Edit the [`config`](./config.yml) as per PrestoDB connection, IAM
    role and other pipeline details such as instance types for various
    pipeline steps etc.

    -   Edit the parameter values in the `presto` section. These
        parameters define the connectivity to PrestoDB.
    -   Edit the parameter values in the `aws` section. These parameters
        define the newtork connectivity, IAM role, bucket name, region
        and other AWS cloud related parameters.
    -   Edit the parameter values in the sections corresponding to the
        pipeline steps i.e. `training_step`, `tuning_step`,
        `transform_step` etc. Review all the parameters in these
        sections carefully and edit them as appropriate for your
        use-case.
    -   Review the parameters in the rest of the sections of the
        [`config`](./config.yml)and edit them if needed.

## Testing the solution

### AWS Architecture

Once the prerequisites are complete and the config.yml file is set up
correctly, we are now ready to run the
[`mlops-pipeline-prestodb`]((https://github.com/aws-samples/mlops-pipeline-prestodb/tree/main))
implementation. Follow the step-by-step walkthrough below as represented
in the architecture diagram. This diagram shows the three portions of
this solution: the training pipeline, batch transform pipeline and
deploying the model as a SageMaker Real-Time Endpoint:

![](images/Architecture_mlops.png)

-   In the first block on the left, we see the architectural
    representation of our training pipeline, which includes all the
    steps that are executed as part of it. This includes the data
    preprocessing step, training and tuning step, model evaluation,
    condition step and lastly the register model step. The train, test,
    validation datasets and the [evaluation
    report](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.html)
    are sent to S3 as a part of this pipeline.
-   In the second block on the right, we see the architectural
    representation of our batch transform pipeline. This includes the
    batch data preprocessing step, approving the latest model from the
    model registry, creating the model and performing batch transform on
    batch data, which is again stored in S3.
-   The PrestoDB server is hosted on an Amazon EC2 instance, with
    credentials stored in [AWS Secrets
    Manager](https://aws.amazon.com/secrets-manager/).
-   Towards the end, the latest approved and trained model from the
    SageMaker Model Registry is deployed as a SageMaker Real-Time
    Endpoint for inference.

### Solution Walkthrough

1.  On the left panel of [SageMaker
    Studio](https://aws.amazon.com/sagemaker/studio/), choose
    **0_model_training_pipeline.inpynb** in the navigation pane. When
    the notebook is open, on the Run menu, choose **Run All Cells** to
    run the code in this notebook. This notebook demonstrates how
    SageMaker Pipelines can be used to string together a sequence of
    data processing, model training, tuning and evaluation step to train
    a binary classification machine learning model using scikit-learn.
    The trained model can then be used for batch inference, or hosted on
    a SageMaker Endpoint for real-time inference. At the end of this
    run, navigate to
    [`pipelines`](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-studio.html)
    on the Studio Navigation pane:

    **After executing the entire training pipeline, your pipeline
    structure on [Amazon SageMaker
    Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html)
    should look like this:**

    ![](images/training_pipeline.png)
    ***View the step-by-step description of the training pipeline as
    follows***:

    1.  **Preprocess data step**: In this step of the notebook, we
        create a processing job for data processing. For more
        information on processing jobs, see [Process
        Data](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html).
        We use a [preprocess
        script](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/presto_preprocess_for_training.py)
        which is used to connect and query data from the PrestoDB
        instance (using the user specified query in the [config
        file](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/config.yml)).
        This step then splits and sends the data as `tain`, `test`, and
        `validation` files to an S3 bucket. Using the data in these
        files, we can train our machine learning model.

        We use the
        [sklearn_processor](https://docs.aws.amazon.com/sagemaker/latest/dg/use-scikit-learn-processing-container.html)
        in our
        [`ProcessingStep`](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing)
        and define it as given below:

        ``` python
        # declare the sk_learn processer
        step_args = sklearn_processor.run(
                ## code refers to the data preprocessing script that is responsible for querying data from the PrestoDB instance
                code=config['scripts']['preprocess_data'],
                source_dir=config['scripts']['source_dir'], 
                outputs=outputs_preprocessor,
                arguments=[
                    "--host", host_parameter,
                    "--port", port_parameter,
                    "--presto_credentials_key", presto_parameter,
                    "--region", region_parameter,
                    "--presto_catalog", presto_catalog_parameter,
                    "--presto_schema", presto_schema_parameter,
                    "--train_split", train_split.to_string(), 
                    "--test_split", test_split.to_string(),
                ],
            )

            step_preprocess_data = ProcessingStep(
                name=config['data_processing_step']['step_name'],
                step_args=step_args,
            )
        ```

        Here, we use the `config['scripts']['source_dir']` which points
        to our data preprocessing script that connects to the PrestoDB
        instance. Parameters being used as arguments in the
        [`step_args`](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#:~:text=%2C%0A%20%20%20%20sagemaker_session%3Dpipeline_session%2C%0A-,step_args,-%3D%20pyspark_processor.run(%0A%20%20%20%20inputs))
        include the `Presto host`, `port`, AWS account information and
        credentials that are configurable via the config file.

    2.  **Train Model Step**: In this step of the pipeline, we create a
        training job to train a model. For more information on training
        jobs, see [Train a Model with Amazon
        SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html).
        Here, we use the [Scikit Learn
        Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html)
        from the SageMaker SDK and the `RandomForestClassifier` to train
        the ML model for our binary classification use case. The
        [HyperparameterTuner](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html)
        class is used for running automatic model tuning to determine
        the set of hyperparameters that provide the best performance
        based on a given metric threshold (for example, maximizing the
        AUC metric).

        -   In the code below, we first use the `sklearn_estimator`
            object with parameters that are configured in the [config
            file](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/config.yml)
            and uses this [training
            script](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/training.py)
            to train the ML model. This step accesses the `train`,
            `test` and `validation` files that are created as a part of
            the previous data preprocessing step and is used in the step
            below:

            ``` python
            # declare a tuning step to use the train and test data to tune the ML model using the `HyperparameterTuner` declared above
            step_tuning = TuningStep(
                name=config['tuning_step']['step_name'],
                tuner=rf_tuner,
                inputs={
                    "train": TrainingInput(
                        s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                            "train" ## refer to this
                        ].S3Output.S3Uri,
                        content_type="text/csv",
                    ),
                    "test": TrainingInput(
                    s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    content_type="text/csv",
                    ),
                },
            )
            ```

    3.  **Evaluate model step**: This step in the pipeline checks if the
        trained and tuned model has an accuracy level above a
        configurable threshold and only then registers the model with
        the Model Registry (from where it can be subsequently [approved
        and
        deployed](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html)).
        If the model accuracy does not meet the[given
        threshold](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.html#Define-a-Model-Evaluation-Step-to-Evaluate-the-Trained-Model)
        then the pipeline fails and the model is not registered with the
        Model Registry. We use the
        [`ScriptProcessor`](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html)
        with an [evaluation
        script](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/evaluate.py)
        that a user creates to evaluate the trained model based on a
        metric of choice.

        -   The evaluation step uses the [evaluation
            script](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/evaluate.py)
            as a code entry in the step below. This script prepares the
            features, target values and calculates the prediction
            probabilities using `model.predict`. An `evaluation report`
            is sent to S3 that contains information on
            `precision, recall, accuracy` metrics.

        ``` python
        step_evaluate_model = ProcessingStep(
            name=config['evaluation_step']['step_name'],
            processor=evaluate_model_processor,
            inputs=[
                ProcessingInput(
                    source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=bucket),
                    destination="/opt/ml/processing/model",
                    input_name="model.tar.gz" 
                ),
                ProcessingInput(
                    source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                    input_name="test.csv" 
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(bucket),
                            prefix,
                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                            "evaluation",
                        ]
                    )
                )
            ],
            code = config['scripts']['evaluation'],
            property_files=[evaluation_report],
            job_arguments=[
                "--target", target_parameter,
                "--features", feature_parameter,
            ]
        )
        ```

        -   From the `TPCH data` queried, we evaluate the `accuracy`,
            `precision` and `recall`. Once this step is complete, we can
            analyze the associated metrics in the `evaluation report`
            that is sent to the S3 bucket as follows:

        ![](images/evaluation_report_example.png)

    4.  **Condition model step**: Once the model is evaluated, we can
        add conditions to the pipeline with a
        [ConditionStep](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html)
        to register the model only if a certain metric threshold is met
        by the evaluated model. In this case, we only want to register
        the new model version with the Model Registry only if the new
        model meets a specific accuracy condition of above 70%.

        ``` python
        # Create a SageMaker Pipelines ConditionStep, using the condition above.
        # Enter the steps to perform if the condition returns True / False.
        step_cond = ConditionStep(
            name=config['condition_step']['step_name'],
            conditions=[cond_gte],
            if_steps=[step_register_model],
            else_steps=[step_fail], ## if this fails
        )
        ```

        If the `accuracy condition` is not met, a `step_fail` step is
        executed that sends an error message to the user and the
        pipeline fails. For instance, since the `accuracy condition` is
        set to `0.7` in the `config file` and our Accuracy calculates
        exceeds it (`73.8% > 70%`), the `outcome` of this step is set to
        `True` and moved to the last step of the training pipeline.

    5.  **Register model step**: This `RegisterModel` step is to
        register a
        [sagemaker.model.Model](https://sagemaker.readthedocs.io/en/stable/api/inference/model.html)
        or a
        [sagemaker.pipeline.PipelineModel](https://sagemaker.readthedocs.io/en/stable/api/inference/pipeline.html#pipelinemodel)
        with the Amazon SageMaker model registry. Once the trained model
        meets the model performance requirements, a new version of the
        model is registered with the [Model
        Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html).

        ***The model is registered with the Model Registry with approval
        status set to `PendingManualApproval`, this means the model
        cannot be deployed on a SageMaker Endpoint unless its status in
        the registry is changed to `Approved` manually via the SageMaker
        console, programmatically or through a Lambda function.***

    6.  **Orchestrate all steps and start the pipeline**: Once you have
        created the pipeline steps above, you can instantiate and
        execute it with custom parameters making the pipeline agnostic
        to who is triggering it, but also to the scripts and data used.
        The pipeline can be started using the CLI, the SageMaker Studio
        UI or the SDK.

        ``` python
        # Start pipeline with credit data and preprocessing script
        execution = pipeline.start(
                        execution_display_name=pipeline.name,
                        parameters=dict(
                        AccuracyConditionThreshold=config['evaluation_step']['accuracy_condition_threshold'],
                        MaximumParallelTrainingJobs=config['tuning_step']['maximum_parallel_training_jobs'],
                        MaximumTrainingJobs=config['tuning_step']['maximum_training_jobs'],
                        ModelGroup=config['register_model_step']['model_group'],
                    ),
                )
        ```

    ***Now that the model is registered, you can get access to the
    registered model manually on the SageMaker studio Model Registry
    console, or programmatically in the next notebook, approve it and
    run the second portion of this solution: Batch Transform Step***

2.  Next Choose
    [`1_batch_transform_pipeline.ipynb`](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/1_batch_transform_pipeline.ipynb).
    When the notebook is open, on the Run menu, choose **Run All Cells**
    to run the code in this notebook. This notebook will run a batch
    transform pipeline using the model trained in the previous notebook.

    **At the end of the batch transform pipeline, your pipeline
    structure on Amazon SageMaker Pipelines should look like this:**

    ![](images/batch_transform_pipeline.png)
    ***View the steps executed above in detail below:***

    1.  **Extract the latest approved model from the SageMaker Model
        Registry**: In this step of the pipeline, we extract the latest
        model from the Model Registry, and set the `ModelApprovalStatus`
        to `Approved`:

        ``` python
        ## updating the latest model package to approved status to use it for batch inference
        model_package_update_response = sm.update_model_package(
            ModelPackageArn=latest_model_package_arn,
            ModelApprovalStatus="Approved",
        )
        ```

        Now we have extracted the latest model from the SageMaker Model
        Registry, and programmatically approved it. You can also approve
        the model manually on the [SageMaker Model
        Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
        page in SageMaker Studio as given in the screenshot below.

        ![](images/sagemaker_model_registry.png)

    2.  **Read raw data for inference from PrestoDB and store in an
        Amazon S3 bucket**: In this step, once the latest model is
        approved, we fetch batch data from the PrestoDB instance and use
        that for our batch transform step. In this step, we use a [batch
        preprocess
        script](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/presto_preprocess_for_batch_inference.py)
        that is responsible for querying data from PrestoDB and saving
        in a batch directory within an S3 bucket. The query used to
        fetch batch data is configured within the config file by the
        user in the `transform_step` section.

        ``` python
        # declare the batch step that is called later in pipeline execution
        batch_data_prep = ProcessingStep(
            name=config['data_processing_step']['step_name'],
            step_args=step_args,
        )
        ```

        Once the batch data is extracted into the S3 bucket, we declare
        a model with an `image uri` and point to the
        [‘inference.py’](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/code/inference.py)
        script that grabs information on features to use while making
        predictions for the trained model. We can then create the model
        as follows:

        ``` python
        # create the model image based on the model data and refer to the inference script as an entry point for batch inference
        model = Model(
            image_uri=image_uri,
            entry_point=config['scripts']['batch_inference'],
            model_data=model_data_url,
            sagemaker_session=pipeline_session,
            role=role,
        )
        ```

    3.  **Create a batch transform step to perform inference on the
        batch data stored in S3**: Now that a model instance is created
        (view above), we create a
        [Transformer](https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html)
        instance with the appropriate model type, compute instance type,
        and desired output S3 URI. Specifically, pass in the `ModelName`
        from the
        [`CreateModelStep`](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html),
        `step_create_model` properties. The `CreateModelStep` properties
        attribute matches the object model of the `DescribeModel`
        response object. A `transform step` for batch transformation is
        used to run inference on an entire dataset. For more information
        about batch transformation, see [Run Batch Transforms with
        Inference
        Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-batch.html).

        -   A transform step requires a transformer and the data on
            which to run batch transformation.

            ``` python
            transformer = Transformer(
            model_name=step_create_model.properties.ModelName,
            instance_type=config['transform_step']['instance_type'],
            instance_count=config['transform_step']['instance_count'],
            strategy="MultiRecord",
            accept="text/csv",
            assemble_with="Line",
            output_path=f"s3://{bucket}",
            tags = config['transform_step']['tags'], 
            env={
                'START_TIME_UTC': st.strftime('%Y-%m-%d %H:%M:%S'), 
                'END_TIME_UTC': et.strftime('%Y-%m-%d %H:%M:%S'),
            })
            ```

        -   Now that our transformer object is created, we pass the
            transformer input (that contains the batch data from our
            `batch preprocess` step) into the `TransformStep`
            declaration:

        ``` python
        step_transform = TransformStep(
            name=config['transform_step']['step_name'], transformer=transformer, inputs=transform_input, 
        )
        ```

3.  Lastly, Choose
    [`2_realtime_inference.ipynb`](https://github.com/aws-samples/mlops-pipeline-prestodb/blob/main/2_realtime_inference.ipynb).
    When the notebook is open, on the Run menu, choose **Run All Cells**
    to run the code in this notebook. This notebook extracts the latest
    approved model from the Model Registry and deploys it as a SageMaker
    Endpoint for real time inference. It does so by executing the
    following steps:

    1.  **Extract the latest approved model from the SageMaker Model
        Registry**: To deploy a real time SageMaker endpoint, first
        fetch the `image uri` and extract the latest approved model the
        same way as done in the batch transform notebook. Once you have
        extracted the latest approved model, use a container list with
        the specified `inference.py` as the script for the deployed
        model to use at inference. This model creation and endpoint
        deployment is specific to the [scikit-learn
        model](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html)
        configuration.

        In this code, we use the `inference.py` file specific to the
        scikit-learn model. We then create our endpoint configuration,
        setting our `ManagedInstanceScaling` to `ENABLED` with our
        desired `MaxInstanceCount` and `MinInstanceCount` for automatic
        scaling:

        ``` python
        create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': instance_type,
            # have max instance count configured here
            'InitialInstanceCount': min_instances,
            'InitialVariantWeight': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic', 
            # change your managed instance configuration here
            "ManagedInstanceScaling":{
                "MaxInstanceCount": max_instances,
                "MinInstanceCount": min_instances,
                "Status": "ENABLED",}
        }])
        ```

    2.  **Run inferences on the deployed real time endpoint**: Once you
        have extracted the latest approved model, created the model from
        the desired `image uri` and configured the
        `Endpoint configuration`, you can then deploy it as a real time
        SageMaker endpoint below:

        ``` python
        create_endpoint_response = sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)

        # wait for endpoint to reach a terminal state (InService) using describe endpoint
        describe_endpoint_response = sm.describe_endpoint(EndpointName=endpoint_name)

        while describe_endpoint_response["EndpointStatus"] == "Creating":
            describe_endpoint_response = sm.describe_endpoint(EndpointName=endpoint_name)
        ```

        Upon deployment, you can view the endpoint in service on the
        SageMaker Endpoints under the Inference option on the left panel
        as follows: <img src="images/ep_in_service.png" id="fig-open-jl"
        alt="SageMaker Endpoint deployed for Real Time Inference" />

    3.  **Now run inference against the data extracted from prestoDB**:

        ``` python
        body_str = "total_extended_price,avg_discount,total_quantity\n1,2,3\n66.77,12,2"

        response = smr.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=body_str.encode('utf-8') ,
            ContentType='text/csv',
        )

        response_str = response["Body"].read().decode()
        response_str
        ```

## Results

Here is a compilation of some queries and responses generated by our
implementation from the real time endpoint deployment stage: \[ to add
results here, querying data, fetching it, making predictions etc\]

<table style="width:50%;">
<caption>mlops-pipeline-prestodb results</caption>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr class="header">
<th>Query</th>
<th>Answer</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>total_extended_price,avg_discount,total_quantity,2,3,12,2</td>
<td>– response –</td>
</tr>
</tbody>
</table>

mlops-pipeline-prestodb results

## Conclusion

We have now seen an end-to-end process of our solution. We fetched data
by connecting a SageMaker Processing Job to a PrestoDB instance,
followed by training, evaluating, registering the model. We then
approved the latest registered model from our training pipeline solution
and ran batch inference against the batch data (queried from PrestoDB)
stored in S3. Furthermore, we deployed the latest approved model as a
real time SageMaker endpoint to run inferences.

With the rise of generative AI, the use of training, deploying and
running machine learning models exponentially increases, and so does the
use of data. With an integration of SageMaker Processing Jobs with
PrestoDB, customers can easily and seamlessly migrate their workloads to
SageMaker pipelines without any burden of additional data preparation,
storage, and access. Customers can now build, train, evaluate, run batch
inferences and deploy their models as real time endpoints while taking
advantage of their existing data engineering pipelines with minimal-no
code changes.

We encourage you to learn more by exploring SageMaker Pipeline,
open-source data querying engines like PrestoDB and building a solution
using the sample implementation provided in this post.

Portions of this code are released under the Apache 2.0 License as
referenced here: https://aws.amazon.com/apache-2-0/

------------------------------------------------------------------------

## Author bio

<img style="float: left; margin: 0 10px 0 0;" src="images/">Amit Arora
is an AI and ML Specialist Architect at Amazon Web Services, helping
enterprise customers use cloud-based machine learning services to
rapidly scale their innovations. He is also an adjunct lecturer in the
MS data science and analytics program at Georgetown University in
Washington D.C.

<br><br>

<img style="float: left; margin: 0 10px 0 0;" src="images/">Madhur
Prashant <br><br>

<br><br>

<img style="float: left; margin: 0 10px 0 0;" src="images/">Antara Raisa
is an AI and ML Solutions Architect at Amazon Web Services supporting
Strategic Customers based out of Dallas, Texas. She also has previous
experience working with large enterprise partners at AWS, where she
worked as a Partner Success Solutions Architect for digital native
customers.
