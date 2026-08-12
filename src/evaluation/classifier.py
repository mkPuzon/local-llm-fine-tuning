import torch
import pandas as pd


def get_text_tokenizer(processor):
    """
    Return the underlying text tokenizer.

    Qwen3VLProcessor contains a tokenizer attribute,
    while ordinary Hugging Face tokenizers can be
    passed through directly.
    """

    if hasattr(processor, "tokenizer"):
        return processor.tokenizer

    return processor


def get_class_token_ids(processor):
    """
    Get the token IDs corresponding to classification
    outputs '0' and '1'.

    Both must be represented by exactly one token.
    """

    tokenizer = get_text_tokenizer(processor)

    zero_ids = tokenizer.encode(
        "0",
        add_special_tokens=False,
    )

    one_ids = tokenizer.encode(
        "1",
        add_special_tokens=False,
    )

    if len(zero_ids) != 1:
        raise ValueError(
            f'"0" is not represented by exactly one token: '
            f"{zero_ids}"
        )

    if len(one_ids) != 1:
        raise ValueError(
            f'"1" is not represented by exactly one token: '
            f"{one_ids}"
        )

    return zero_ids[0], one_ids[0]


def build_prompt(example, system_prompt):
    """
    Construct the prompt presented to the model.

    IMPORTANT:
    This must reproduce the prompt format used during
    training.
    """

    return (
        f"{system_prompt}\n"
        f"User query: {example['query']}\n"
        "Classification: "
    )


def run_inference(
    model,
    processor,
    dataset,
    system_prompt,
    batch_size=8,
):
    """
    Run the fine-tuned classifier over a dataset.

    Returns
    -------
    pandas.DataFrame

        One row per example containing:

        id
        query
        true_label
        probability_1
        logit_0
        logit_1
        source
        tag
    """

    model.eval()

    # --------------------------------------------------------
    # Determine model device
    # --------------------------------------------------------

    device = next(
        model.parameters()
    ).device

    # --------------------------------------------------------
    # Find classification token IDs
    # --------------------------------------------------------

    zero_token_id, one_token_id = (
        get_class_token_ids(processor)
    )

    # --------------------------------------------------------
    # Construct prompts
    # --------------------------------------------------------

    examples = []

    for example in dataset:

        examples.append({
            "id": example.get("id"),
            "query": example["query"],
            "true_label": int(example["label"]),
            "source": example.get("source"),
            "tag": example.get("tag"),
            "prompt": build_prompt(
                example,
                system_prompt,
            ),
        })

    predictions = []

    # --------------------------------------------------------
    # Run batched inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for start in range(
            0,
            len(examples),
            batch_size,
        ):

            batch = examples[
                start:start + batch_size
            ]

            prompts = [
                example["prompt"]
                for example in batch
            ]

            # Qwen3VLProcessor
            inputs = processor(
                text=prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)

            # ------------------------------------------------
            # Find the final real token in each sequence
            # ------------------------------------------------

            sequence_lengths = (
                inputs["attention_mask"].sum(dim=1) - 1
            )

            batch_indices = torch.arange(
                len(batch),
                device=device,
            )

            final_logits = outputs.logits[
                batch_indices,
                sequence_lengths,
            ]

            # ------------------------------------------------
            # Extract only logits for 0 and 1
            # ------------------------------------------------

            logit_0 = final_logits[
                :,
                zero_token_id,
            ]

            logit_1 = final_logits[
                :,
                one_token_id,
            ]

            # ------------------------------------------------
            # Convert relative logits to P(class=1)
            # ------------------------------------------------

            probability_1 = torch.sigmoid(
                logit_1 - logit_0
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            for i, example in enumerate(batch):

                predictions.append({
                    "id": example["id"],
                    "query": example["query"],
                    "true_label": example["true_label"],
                    "probability_1": float(
                        probability_1[i].item()
                    ),
                    "logit_0": float(
                        logit_0[i].item()
                    ),
                    "logit_1": float(
                        logit_1[i].item()
                    ),
                    "source": example["source"],
                    "tag": example["tag"],
                })

    return pd.DataFrame(predictions)