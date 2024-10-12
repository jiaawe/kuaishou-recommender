import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def translate_text(text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': '''You are a translator. Paraphrase the text from Chinese to English.
                    I do not want any Chinese characters.
                    Just give me the translated text ONLY.''',
                },
                {
                    'role': 'user',
                    'content': f'Translate the following Chinese text to English: {text}',
                }
            ],
            model=model, 
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating text {text}: {e}")
        return None

def main():
    global client, model

    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )
    model = 'qwen2.5'

    rootpath = "KuaiRec 2.0/"
    captions = pd.read_csv(rootpath + "data/kuairec_caption_category.csv", engine='python')

    # Test translation
    test_translation = translate_text('精神小伙路难走 程哥你狗粮慢点撒?')
    print(f"Test translation: {test_translation}")

    tqdm.pandas(desc="Translating captions")
    captions['english_caption'] = captions['caption'].progress_apply(translate_text)
    captions.to_csv(rootpath + "data/kuairec_caption_category_translated.csv", index=False)

    tqdm.pandas(desc="Translating category names")
    captions['english_first_level_category_name'] = captions['first_level_category_name'].progress_apply(translate_text)
    captions.to_csv(rootpath + "data/kuairec_caption_category_translated.csv", index=False)

    tqdm.pandas(desc="Translationg second-level category names")
    captions['english_second_level_category_name'] = captions['second_level_category_name'].progress_apply(translate_text)
    captions.to_csv(rootpath + "data/kuairec_caption_category_translated.csv", index=False)

    tqdm.pandas(desc="Translationg third-level category names")
    captions['english_third_level_category_name'] = captions['third_level_category_name'].progress_apply(translate_text)
    captions.to_csv(rootpath + "data/kuairec_caption_category_translated.csv", index=False)

    tqdm.pandas(desc="Translating topic tag")
    captions['english_topic_tag'] = captions['topic_tag'].progress_apply(translate_text)
    captions.to_csv(rootpath + "data/kuairec_caption_category_translated.csv", index=False)

    print("Translation completed. Results saved to kuairec_caption_category_translated.csv")

if __name__ == "__main__":
    main()