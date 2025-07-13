import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings("ignore")

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(
    "qnguyen3/nanoLLaVA",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("qnguyen3/nanoLLaVA", trust_remote_code=True)

prompt = "Answer yes if this image has action, and no if it doens't."

messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(text)

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split("<image>")]
input_ids = torch.tensor(
    text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long
).unsqueeze(0)

image = Image.open("1.jpg")
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

output_ids = model.generate(
    input_ids, images=image_tensor, max_new_tokens=2048, use_cache=True
)[0]

print(
    tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True).strip()
)
