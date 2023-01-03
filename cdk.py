import os
from pathlib import Path
from constructs import Construct
from aws_cdk import App, Stack, Environment, Duration, CfnOutput
from aws_cdk import Environment, Stack
from aws_cdk.aws_lambda import DockerImageFunction, DockerImageCode, FunctionUrlAuthType

# Environment
# CDK_DEFAULT_ACCOUNT and CDK_DEFAULT_REGION are set based on the
# AWS profile specified using the --profile option.
my_environment = Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"])

class SpotifyRecSysDemo(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ##############################
        #       Lambda Function      #
        ##############################
        lambda_fn = DockerImageFunction(
            self,
            "AssetFunction",
            code=DockerImageCode.from_image_asset(str(Path.cwd()), file="Dockerfile"),
            memory_size=1024,
            timeout=Duration.minutes(2),
            environment={
                "SPOTIPY_USER": os.environ["SPOTIPY_USER"],
                "SPOTIPY_REDIRECT_URI": os.environ["SPOTIPY_REDIRECT_URI"],
                "SPOTIPY_CLIENT_ID": os.environ["SPOTIPY_CLIENT_ID"],
                "SPOTIPY_CLIENT_SECRET": os.environ["SPOTIPY_CLIENT_SECRET"],
            },
        )
        # Add HTTPS URL to access Gradio app via browser
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)
        CfnOutput(self, "functionUrl", value=fn_url.url)

app = App()
rust_lambda = SpotifyRecSysDemo(app, "SpotifyRecSysDemo", env=my_environment)

app.synth()