class Template:
    """Base prompt template interface used to encode and verbalize task examples."""
    def encode(self, sample):
        """Return prompted version of the example (without the answer/candidate)"""
        raise NotImplementedError

    def verbalize(self, sample, candidate):
        """Return the prompted version of the example (with the answer/candidate)"""
        return candidate

    def encode_sfc(self, sample):
        """Same as encode, but for SFC (calibration) -- this usually means the input is not included"""
        return "<mask>"

    def verbalize_sfc(self, sample, candidate):
        """Same as verbalize, but for SFC (calibration) -- this usually means the input is not included"""
        return candidate


class RottenTomatoesTemplate(Template):
    """Prompt template for the Rotten Tomatoes task."""
    verbalizer = {0: "bad", 1: "great"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        text = sample.data["text"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        text = sample.data["text"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f" It was {self.verbalizer[candidate]}"


class IMDBTemplate(Template):
    """Prompt template for the IMDB task."""
    verbalizer = {0: "bad", 1: "great"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        text = sample.data["text"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        text = sample.data["text"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f" It was {self.verbalizer[candidate]}"


class RTPolarityTemplate(Template):
    """Prompt template for the RT-Polarity task."""
    verbalizer = {0: "bad", 1: "great"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        text = sample.data["inputs"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        text = sample.data["inputs"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f" It was {self.verbalizer[candidate]}"


class SST2Template(Template):
    """Prompt template for the SST-2 task."""
    verbalizer = {0: "bad", 1: "great"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f" It was {self.verbalizer[candidate]}"


class ColaTemplate(Template):
    """Prompt template for the Cola task."""
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        text = sample.data["sentence"].strip()
        return f"{text}\nIs the sentence grammatically acceptable? Answer:\n"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        text = sample.data["sentence"].strip()
        return (
            f"{text}\nIs the sentence grammatically acceptable?"
            f" Answer:\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f" Is this sentence acceptable?"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f" Is this sentence acceptable? {self.verbalizer[candidate]}"


class CopaTemplate(Template):
    """Prompt template for the COPA task."""
    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def get_conjucture(self, sample):
        """Choose the conjunction that matches the COPA question type."""
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        """Construct the base COPA prompt before a candidate answer is inserted."""
        premise = sample.data["premise"].rstrip()
        if premise.endswith(
            "."
        ):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        """Adjust candidate casing to match the selected COPA prompt style."""
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class BoolQTemplate(Template):

    """Prompt template for the BoolQ task."""
    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {candidate}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return ""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return candidate


class BoolQTemplateV2(Template):
    """Prompt template for the BoolQ V2 task."""

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n{candidate}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return ""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return candidate


class BoolQTemplateV3(Template):
    """Prompt template for the BoolQ V3 task."""

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n{candidate}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return ""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return candidate


class MultiRCTemplate(Template):
    # From PromptSource 1
    """Prompt template for the MultiRC task."""
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return (
            f'{paragraph}\nQuestion: {question}\nI found this answer "{answer}". Is'
            " that correct? Yes or No?\n"
        )

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return (
            f'{paragraph}\nQuestion: {question}\nI found this answer "{answer}". Is'
            f" that correct? Yes or No?\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"{self.verbalizer[candidate]}"


class CBTemplate(Template):
  # From PromptSource 1
    """Prompt template for the CB task."""
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return (
            f'Suppose {premise} Can we infer that "{hypothesis}"? Yes, No, or'
            " Maybe?\n"
        )

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return (
            f'Suppose {premise} Can we infer that "{hypothesis}"? Yes, No, or'
            f" Maybe?\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"{self.verbalizer[candidate]}"


class WICTemplate(Template):
  # From PromptSource 1
    """Prompt template for the WiC task."""
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return (
            f'Does the word "{word}" have the same meaning in these two sentences?'
            f" Yes, No?\n{sent1}\n{sent2}\n"
        )

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return (
            f'Does the word "{word}" have the same meaning in these two sentences?'
            f" Yes, No?\n{sent1}\n{sent2}\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"{self.verbalizer[candidate]}"


class WSCTemplate(Template):
    # From PromptSource 1
    """Prompt template for the WSC task."""
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        text = sample.data["text"]
        span1 = sample.data["span1_text"]
        span2 = sample.data["span2_text"]
        return (
            f'{text}\nIn the previous sentence, does the pronoun "{span2.lower()}"'
            f" refer to {span1}? Yes or No?\n"
        )

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        text = sample.data["text"]
        span1 = sample.data["span1_text"]
        span2 = sample.data["span2_text"]
        return (
            f'{text}\nIn the previous sentence, does the pronoun "{span2.lower()}"'
            f" refer to {span1}? Yes or No?\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"{self.verbalizer[candidate]}"


class ReCoRDTemplate(Template):
    # From PromptSource 1 but modified

    """Prompt template for the ReCoRD task."""
    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        passage = sample.data["passage"]
        query = sample.data["query"]
        return f'{passage}\n{query}\nQuestion: what is the "@placeholder"\nAnswer:'

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        passage = sample.data["passage"]
        query = sample.data["query"]
        return (
            f'{passage}\n{query}\nQuestion: what is the "@placeholder"\nAnswer:'
            f" {candidate}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f"Answer:"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"Answer: {candidate}"


class ReCoRDTemplateGPT3(Template):
    """Prompt template for the ReCoRD GPT3-style task."""
    # From PromptSource 1 but modified

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        passage = sample.data["passage"].replace("@highlight\n", "- ")
        return f"{passage}\n-"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        passage = sample.data["passage"].replace("@highlight\n", "- ")
        query = sample.data["query"].replace(
            "@placeholder",
            candidate[0] if isinstance(candidate, list) else candidate,
        )
        return f"{passage}\n- {query}"

        # passage = sample.data['passage']
        # query = sample.data['query']
        # return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f"-"

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        query = sample.data["query"].replace(
            "@placeholder",
            candidate[0] if isinstance(candidate, list) else candidate,
        )
        return f"- {query}"


class RTETemplate(Template):
    # From PromptSource 1
    """Prompt template for the RTE task."""
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        premise = sample.data["sentence1"]
        hypothesis = sample.data["sentence2"]
        return (
            f'{premise}\nDoes this mean that "{hypothesis}" is true? Yes or No?\n'
        )

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        premise = sample.data["sentence1"]
        hypothesis = sample.data["sentence2"]
        return (
            f'{premise}\nDoes this mean that "{hypothesis}" is true? Yes or'
            f" No?\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"{self.verbalizer[candidate]}"


class TwitterEmotionTemplate(Template):
    # From PromptSource 1
    """Prompt template for the Twitter Emotion task."""
    verbalizer = {0: "sadness", 1: "joy"}

    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        return f'{sample.data["text"]} Does the tweet express joy or sadness?\n'

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        return (
            f'{sample.data["text"]} Does the tweet express joy or'
            f" sadness?\n{self.verbalizer[candidate]}"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        return f""

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        return f"{self.verbalizer[candidate]}"


class SQuADv2Template(Template):

    """Prompt template for the SQuAD V2 task."""
    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        question = sample.data["question"].strip()
        title = sample.data["title"]
        context = sample.data["context"]
        answer = sample.data["answers"][
            0
        ]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        question = sample.data["question"].strip()
        title = sample.data["title"]
        context = sample.data["context"]
        answer = sample.data["answers"][
            0
        ]  # there are multiple answers. for the prompt we only take the first one

        return (
            f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"
            f" {answer}\n"
        )

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        raise NotImplementedError


class DROPTemplate(Template):

    """Prompt template for the DROP task."""
    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        question = sample.data["question"].strip()
        # title = sample.data['title']
        context = sample.data["context"]
        answer = sample.data["answers"][
            0
        ]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        question = sample.data["question"].strip()
        # title = sample.data['title']
        context = sample.data["context"]
        answer = sample.data["answers"][
            0
        ]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        raise NotImplementedError


class GigaTemplate(Template):

    """Prompt template for the Gigaword task."""
    def encode(self, sample):
        """Render the task input without attaching an answer option."""
        document = sample.data["document"].strip()
        return f"{document} Please summarize the sentence. Answer:"

    def verbalize(self, sample, candidate):
        """Render the task input with the provided answer option attached."""
        document = sample.data["document"].strip()
        answer = sample.data["summary"].strip()
        return f"{document} Please summarize the sentence. Answer: {answer}"

    def encode_sfc(self, sample):
        """Render the calibration-only prompt used for surface-form competition."""
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        """Render the calibration prompt with a candidate answer attached."""
        raise NotImplementedError
