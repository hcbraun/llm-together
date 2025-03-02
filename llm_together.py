import llm
import together
from pydantic import Field
from typing import Optional

@llm.hookimpl
def register_models(register):
    models = Together().client.models.list()
    for model in sorted(models, key=lambda m: m.id):
        register(Together(model))


def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}

class Together(llm.KeyModel):
    model_id = "llm-together"
    needs_key = "together"
    key_env_var = "TOGETHER_API_KEY"
    default_stop = ["<human>"]
    can_stream = True

    def __init__(self, model=None):
        self.client = together.Client(api_key=self.get_key())
        if model:
            self.model = model
            self.model_id = model.id

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "What sampling temperature to use, between 0 and 2. Higher values like "
                "0.8 will make the output more random, while lower values like 0.2 will "
                "make it more focused and deterministic."
            ),
            ge=0,
            le=2,
            default=None,
        )
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate.", default=8192
        )
        top_p: Optional[float] = Field(
            description=(
                "An alternative to sampling with temperature, called nucleus sampling, "
                "where the model considers the results of the tokens with top_p "
                "probability mass. So 0.1 means only the tokens comprising the top "
                "10% probability mass are considered. Recommended to use top_p or "
                "temperature but not both."
            ),
            ge=0,
            le=1,
            default=None,
        )
        repetition_penalty: Optional[float] = Field(
            description=(
                "A number that controls the diversity of generated text by "
                "reducing the likelihood of repeated sequences. Higher values "
                "decrease repetition."
            ),
            ge=-2,
            le=2,
            default=None,
        )

    def execute(self, prompt, stream, response, conversation, key=None):
        kwargs = dict(not_nulls(prompt.options))

        user_prompt = "{}\n\n{}".format(prompt.system or "", prompt.prompt)
        history = ""
        stop = self.default_stop

        if 'config' in self.model:
            if conversation is not None:
                for message in conversation.responses:
                    if 'prompt_format' in self.model["config"] and self.model["config"]['prompt_format']:
                        formatted_prompt = self.model["config"]["prompt_format"].format(prompt=message.prompt)
                        message_text = message.text()
                        history += formatted_prompt + " " + message_text + "\n"
                    else:
                        history += "{}\n\n{}".format(message.prompt, message.text())+ "\n"

            if 'prompt_format' in self.model["config"] and self.model["config"]['prompt_format']:
                user_prompt = self.model["config"]["prompt_format"].format(prompt = user_prompt)


            if 'stop' in self.model["config"]:
                config_stop = self.model["config"]["stop"]
                if isinstance(config_stop, list):
                    stop = config_stop
                else:
                    stop = [config_stop]

        if stream:
            for chunk in self.client.completions.create(
                prompt =  history + "\n" + user_prompt,
                model = self.model_id,
                stream = True,
                stop = stop,
                **kwargs,
            ):
                if chunk.choices and len(chunk.choices) > 0:
                    yield chunk.choices[0].text
        else:
            output = self.client.completions.create(
                prompt =  history + "\n" + user_prompt,
                model = self.model_id,
                stop = stop,
                **kwargs,
            )
            if output.choices and len(output.choices) > 0:
                yield output.choices[0].text
