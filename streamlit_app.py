import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def build_prompt(ingredients: str, vegetarian: bool) -> str:
    veg_clause = "Make vegetarian versions when possible." if vegetarian else "Include an optional healthier or vegetarian version when applicable."
    prompt = (
        "You are a helpful chef. Create 3 recipe ideas using only these ingredients: "
        f"{ingredients}. For each recipe, provide:\n- Recipe name\n- Ingredients with approximate quantities\n- Step-by-step cooking instructions\n- Estimated total cooking time\n- An optional healthier or vegetarian version\n{veg_clause}\nKeep answers concise and easy to follow."
    )
    return prompt


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    st.title("AI Recipe Generator")
    st.write("Enter ingredients you have and get recipe ideas using google/flan-t5-base.")

    ingredients = st.text_area("Ingredients (comma separated)", height=80)
    vegetarian = st.checkbox("Prefer vegetarian/healthy suggestions", value=False)
    max_tokens = st.slider("Max tokens (response length)", 64, 1024, 256)

    if st.button("Generate Recipes"):
        if not ingredients.strip():
            st.error("Please enter some ingredients.")
            return

        prompt = build_prompt(ingredients, vegetarian)
        with st.spinner("Generating recipes—this may take a minute the first time..."):
            try:
                result = generate(prompt, max_new_tokens=max_tokens)
            except Exception as e:
                st.error(f"Error generating recipes: {e}")
                return

        st.subheader("Generated Recipes")
        st.code(result)

        st.markdown("---")
        st.write("You can refine results by editing the ingredients or toggling the vegetarian option.")


if __name__ == "__main__":
    main()
