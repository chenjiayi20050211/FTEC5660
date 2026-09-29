# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
This solution builds a multimodal LangChain pipeline to parse supermarket receipt images and compute the aggregated monetary values required by the two questions.

In `build_chain()`, I construct the inference chain using the required vision model `deepseek-v4-flash-vision-exp`. A carefully designed prompt instructs the vision model to extract exactly three structured values from each receipt image and return pure JSON output:
- `final_payment`: the final paid amount after receipt rounding adjustment
- `subtotal`: subtotal amount before rounding
- `total_discount`: sum of all promotions, coupons, membership and app discounts. Rounding correction is excluded; negative discount values shown on receipts should be converted to positive.

`JsonOutputParser` is used to parse the JSON output from the LLM into Python dictionary. The full chain is composed as `prompt | llm | JsonOutputParser`.

In `answer_queries()`, the provided `image_data_url()` helper encodes each receipt image to the base64 data URL format required for multimodal input. `chain.batch()` is applied to run parallel inference across all receipt images for better performance.

All monetary calculations use Python `Decimal` type to avoid floating-point precision errors. I take absolute value of `total_discount` to handle negative discount notations on receipts. Exception handling is implemented: if parsing fails for a receipt, all three values fall back to `0.00`.

For aggregation:
1. Query 1 (total spent): sum all `final_payment` values from all receipts.
2. Query 2 (amount without discount): sum `subtotal + total_discount` for every receipt, which adds back all discounts while ignoring rounding differences, following the homework definition.

The final responses are formatted as `HK$XX.XX`. Each returned string contains exactly one HKD amount, satisfying the regex rule in the grader. No hardcoded answers or filenames are used; the pipeline works for any folder of receipt images as specified.
> to students: please fill your solution description here.

### Chain Visualization
```mermaid
flowchart LR
    A[Input: receipt images folder] --> B[Encode images via image_data_url]
    B --> C[ChatPromptTemplate<br/>Instruction: extract 3 fields JSON]
    C --> D[DeepSeek VLM deepseek-v4-flash-vision-exp]
    D --> E[JsonOutputParser]
    E --> F[Batch parallel inference for all receipts]
    F --> G[For each receipt: compute with Decimal<br/>final_payment, subtotal, total_discount]
    G --> H[Aggregate sum<br/>Q1: sum final_payment<br/>Q2: sum(subtotal+discount)]
    H --> I[Return HK$xx.xx response]
    I --> J[Write results.csv]

