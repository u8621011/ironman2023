# 各類引入
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, TransformChain

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.document_loaders import SRTLoader
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.vectorstore import VectorStore
import re


model_name = 'gpt-3.5-turbo-16k'
temperature = 0
verbose = False # for debug
default_youtube_video_file = 'ironman2023/srt_files/SULAWESI _ Makassar & Malino _ Indonesia Travel VLOG 1 - YouTube - English.srt'

llm_chat = None
memory = None

def init_env():
    """
    初始所有的環境設定

    """
    global llm_chat, memory

    # 這裏是我們的大腦
    llm_chat = ChatOpenAI(temperature=0)

    # memory 就像是我們的記憶
    memory = ConversationBufferMemory(return_messages=True, input_key="input")  # 預設 memory_key 為 history


def load_document_from_srt_file(srt_file):
    loader = SRTLoader(srt_file)

    document_loaded = loader.load()
    content_lines = document_loaded[0].page_content.splitlines()

    return content_lines



#================================================================================================
# 以下是取得影片內容階段的各類提示設計
#================================================================================================


######################################
# 「取得學習主題」的提示設計
######################################
def get_topic_extractor_chain():
    """
    這裏是取得語言學習情景的任務器
    """
    topic_extractor_template = """你是一個使用者語言學習情景的蒐集機器人，
    所謂的學習情景例如，旅遊、商業用語、生活用語等。
    詳細點的，例如，旅遊中的機場、飯店、餐廳、交通工具等也是。

    你的工作則是從使用者的訊息中，提取出使用者指明的學習情景，並且輸出那個情景。

    請記得你只會輸出提取到學習情境，不要輸出其他的訊息。如果沒有找到任何學習情景的資訊時，你只能回傳 None。
    """

    system_message_prompt_topic = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = topic_extractor_template,
                input_variables=[],
            )
        )

    chat_prompt_topic_extractor = ChatPromptTemplate.from_messages([
        system_message_prompt_topic,
        HumanMessagePromptTemplate.from_template('我想學約會時可以用到的一些詞彙或者句子'),
        AIMessagePromptTemplate.from_template('約會時的情景用語'),
        HumanMessagePromptTemplate.from_template('{input}')
    ])

    chain = LLMChain(llm=llm_chat, prompt=chat_prompt_topic_extractor, verbose=verbose, output_key='learning_topic')

    return chain

topic_extractor_chain = get_topic_extractor_chain()


######################################
# 「從學習主題生成例句」的提示設計
######################################
def get_sample_sentence_generation_chain():
    tempalte = """你是一個例句生成器，你的任務是根據使用者提供給你的語言使用情景，生成以下條件的的常用例句。

    生成條件
    生成目標語言：{learning_lang}
    例句數量： 1 句
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(tempalte)

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="語言使用情景: {learning_topic}",
            input_variables=["learning_topic"],
        )
    )

    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(
        llm=llm_chat,
        prompt=chat_prompt_template,
        output_key="sample_sentence")

    return chain


sentence_generation_chain = get_sample_sentence_generation_chain()


######################################
# 「搜尋學習主題影片」的提示設計
######################################
def search_video_from_sample_sentence_func(inputs: dict) -> dict:
    sample_sentence = inputs["sample_sentence"]
    db_vector = inputs["db_vector"]
    docs_searched = db_vector.similarity_search(sample_sentence)

    return {"similar_sentence": docs_searched[0].page_content,
            "video_source": docs_searched[0].metadata["source"]
            }

topic_video_search_chain = TransformChain(
    input_variables=["sample_sentence"],
    output_variables=["similar_sentence", "video_source"],
    transform=search_video_from_sample_sentence_func
)


######################################
# 「讀取學習主題影片」提示設計
######################################
def load_document_from_video_source(inputs: dict) -> dict:
    video_source_full = 'ironman2023/' + inputs["video_source"]

    document_loaded = load_document_from_srt_file(video_source_full)
    return {"document_loaded": document_loaded}

document_loader_from_video_source_chain = TransformChain(
    input_variables=["video_source"],
    output_variables=["document_loaded"],
    transform=load_document_from_video_source
)


######################################
# 「回應生成」的提示設計
######################################
def get_learning_mode_guide_chain():
    template = """我們是一個透過 youtube影片內容來學習語言的線上學習系統，我們已經透過使用者的訊息選擇了接下來的學習影片，請你接下來引導使用者選擇他想要的學習方式。

    背景資料：
    使用者的母語： {user_lang}
    使用者想學習的語言： {learning_lang}
    使用者選擇的學習影片： {video_source}

    我們提供的學習方式有：
    1. 影片內容摘要： 讓使用者可以快速了解影片內容
    2. 詞彙學習： 使用影片內使用過的詞彙來學習
    3. 文法學習： 使用影片內的例句來做文法學習
    4. 延伸詞彙學習： 任何一個使用者提出的詞彙的相關詞介紹
    """

    system_message_prompt_default = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = template,
                input_variables=["user_lang", "learning_lang", "video_source"],
            )
        )

    chat_prompt_template = ChatPromptTemplate.from_messages([
        system_message_prompt_default,
        MessagesPlaceholder(variable_name="history"),  # key in memory
        HumanMessagePromptTemplate.from_template('{input}')
    ])

    chain = LLMChain(llm=llm_chat, prompt=chat_prompt_template, memory=memory, verbose=verbose)

    return chain

learning_mode_guide_chain = get_learning_mode_guide_chain()


######################################
# 「從學習主題讀取文件」的集成執行鏈提示設計
######################################
topic_document_loading_overall_chain = SequentialChain(
    chains=[
        topic_extractor_chain,
        sentence_generation_chain,
        topic_video_search_chain,
        document_loader_from_video_source_chain,
        learning_mode_guide_chain
    ],
    input_variables=[
        # topic_extractor_chain 需要的變數
        'input',
        # learning_mode_guide_chain 需要的變數
        "user_lang",
        "learning_lang",
    ],
    # Here we return multiple variables
    output_variables=[
        # topic_extractor_chain 的輸出
        "learning_topic",
        # sentence_generation_chain 的輸出
        "sample_sentence",
        # topic_video_search_chain 的輸出
        "similar_sentence",
        "video_source",
        # document_loader_from_video_source_chain 的輸出
        "document_loaded",
        # learning_mode_guide_chain 的輸出
        "text",
    ],
    verbose=verbose)


######################################
# 「YouTube 網址提取器」的提示設計
######################################
def get_youtube_url_extractor_chain():
    youtube_url_extractor_template = """你是一個專門提取YouTube網址的提取器。

    當訊息中有YouTube網址時，你將返回該網址。
    若訊息中沒有YouTube網址，你只能回應「None」。

    除此之外，不回應任何指示、解釋或其他信息。
    """

    system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = youtube_url_extractor_template,
                input_variables=[],
            )
        )

    human_message_prompt = HumanMessagePromptTemplate.from_template('訊息: {input}')

    chat_prompt_extractor = ChatPromptTemplate.from_messages([
        system_message_prompt,
        HumanMessagePromptTemplate.from_template('訊息: 我想學這個影片內的用語： https://www.youtube.com/watch?v=9C06ZPF8Uuc'),
        AIMessagePromptTemplate.from_template('https://www.youtube.com/watch?v=9C06ZPF8Uuc'),
        human_message_prompt
    ])

    chain = LLMChain(llm=llm_chat, prompt=chat_prompt_extractor, verbose=verbose, output_key='youtube_url')

    return chain

youtube_url_extractor_chain = get_youtube_url_extractor_chain()


######################################
# 「Youtube URL 文件讀取回應生成」的提示設計
######################################
def is_valid_youtube_url(url):
    # 定義 YouTube 影片網址的正則表達式
    pattern = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([\w-]+)(?:\S+)?$"

    # 使用 re 模組的 match 函式進行匹配
    match = re.match(pattern, url)

    # 如果匹配成功且影片 ID 不為空，則判斷為合法的 YouTube 影片網址
    if match and match.group(1):
        return True
    else:
        return False
    

def get_video_guide_chain():
    # default chain
    template = """你是一個透過youtube影片內容來學習語言的線上學習平台業務人員，基於以下背景資料以及工作目標來回覆使用者訊息。

    背景資料
    主要的溝通語言： {user_lang}
    使用者想要學習的語言： {learning_lang}
    我們影片的學習方式： 從影片的內容來從中學習詞彙以及例句或者文法。

    工作目標
    引導使用者設定學習語言的影片。
    在正式教導使用者語言前，我們需要一個教學影片，這個就是你要引導設定的原因，而我們設定影片的方式有兩種：一個是透過使用者直接指定想要學習的 youtube 影片網址，另外一種就是使用他們想要學習的主題來搜尋我們的影片庫，
    請你引導他們往這兩個方向來回答他們的需求。
    """

    system_message_prompt_default = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = template,
                input_variables=['user_lang', 'learning_lang'],
            )
        )

    chat_prompt_default = ChatPromptTemplate.from_messages([
        system_message_prompt_default,
        MessagesPlaceholder(variable_name="history"),  # key in memory
        HumanMessagePromptTemplate.from_template('{input}')
    ])


    chain = LLMChain(llm=llm_chat, prompt=chat_prompt_default, memory=memory, verbose=verbose, output_key="text")

    return chain

video_guide_chain = get_video_guide_chain()


def process_youtube_url_extract_result(inputs: dict) -> dict:
    """
    我們這裏有幾個任務。

    1. 確保真的取得正式的 youtube url -> 讀取預設的 youtube 字幕檔 & 送到下個模式的引導器來做回覆
    2. 無效的 youtube url -> 使用這個階段的引導器做回覆
    """
    youtube_url = inputs["youtube_url"]
    video_guide_responser = inputs["video_guide_responser"]
    learning_mode_responser = inputs["learning_mode_responser"]


    if is_valid_youtube_url(youtube_url):
        loaded_document = load_document_from_srt_file(default_youtube_video_file)

        print(f'calling {learning_mode_responser}')
        inputs['video_source'] = youtube_url
        leanring_mode_guide_response = learning_mode_responser(inputs)

        return {
            "document_loaded": loaded_document,
            "text": leanring_mode_guide_response['text']
        }
    else:
        video_guide_response = video_guide_responser(inputs)
        return {
            "document_loaded": None,
            "text": video_guide_response['text']
        }

youtube_url_process_chain = TransformChain(
    input_variables=["youtube_url", "video_guide_responser", "learning_mode_responser"],
    output_variables=["document_loaded", "text"],
    transform=process_youtube_url_extract_result
)


######################################
# Youtube URL 文件讀取的集成執行鏈」提示設計
######################################
youtube_url_document_loading_overall_chain = SequentialChain(
    chains=[
        youtube_url_extractor_chain,
        youtube_url_process_chain
    ],
    input_variables=[
        # youtube_url_extractor_chain 需要的變數
        'input',
        # youtube_url_process_chain 需要的變數
        "user_lang",
        "learning_lang",
        "video_guide_responser",
        'learning_mode_responser'
    ],
    # Here we return multiple variables
    output_variables=[
        # youtube_url_extractor_chain 的輸出
        "youtube_url",
        # youtube_url_process_chain 的輸出
        "document_loaded",
        "text",
    ],
    verbose=verbose)


######################################
# 讀取文件的路由執行鏈設計
######################################
def get_vidoe_guide_router_chain():
    # setup for LLMRouterChain
    prompt_infos = [
        {
            "name": "Topic extractor",
            "description": "適用於取得使用者的學習主題。",
            "chain": topic_document_loading_overall_chain,
        },
        {
            "name": "YoutubeURL extractor",
            "description": "使用於取得使用者訊息中的 YouTube 影片網址。",
            "chain": youtube_url_document_loading_overall_chain
        },
    ]

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm_chat, router_prompt)

    return router_chain

video_guide_router_chain = get_vidoe_guide_router_chain()


######################################
# 取得學習影片階段的 routering handling 函式
######################################
def video_guide_phase_handler(user_lang: str, learning_lang: str, db_vector: VectorStore, user_input: str) -> str:
    global learning_document

    # 訊息的路由判斷
    router_result = video_guide_router_chain(user_input)

    if verbose:
        print(f'router_result: {router_result}')

    ai_response = None
    if router_result['destination'] == 'Topic extractor':
        if verbose:
            print(f'從指定主題載入學習文件')

        # 根據學習主題載入文件
        ai_response = topic_document_loading_overall_chain({
            "learning_lang": learning_lang,
            "user_lang": user_lang,
            'db_vector': db_vector,
            "input": user_input
        })

        learning_document = ai_response['document_loaded']

        if verbose:
            print('learning_document:', learning_document)
    elif router_result['destination'] == 'YoutubeURL extractor':
        if verbose:
            print(f'從預設的 srt 檔案載入學習文件')

        ai_response = youtube_url_document_loading_overall_chain({
            'input': user_input,
            'user_lang': user_lang,
            'learning_lang': learning_lang,
            'video_guide_responser': video_guide_chain,
            'learning_mode_responser': learning_mode_guide_chain,
        })

        learning_document = ai_response['document_loaded']

        if verbose:
            print('learning_document:', learning_document)
    else:
        # 預設處理流程
        ai_response = video_guide_chain({
            'user_lang': user_lang,
            'learning_lang': learning_lang,
            'input': user_input,
        })

    return ai_response["text"]


#================================================================================================
# 以下是學習模式階段的各類提示設計
#================================================================================================

######################################
# 摘要學習
######################################
def get_video_digest_chain():
    template="""你是一個影片內容摘要器，請將下方的影片內容，以下面條件做簡易的摘要介紹。

    摘要語言： {learning_lang}
    影片內容：{video_content}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     output_key="video_digest",
                     verbose=verbose)

    return chain

video_digest_chain = get_video_digest_chain()


def get_digest_teaching_chain():
    template="""你是一個專業的外語老師也是一個說故事高手。我會提供給你一個影片的簡易摘要，
    你要以親切並且面對面的口吻幫學生做個影片的導讀。

    你的導讀內容最好也要包含學生正在學習的語言 以及學生的母語。

    學生學習的語言： {learning_lang}
    學生的母語： {user_lang}
    影片簡易摘要： {video_digest}
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["learning_lang", "user_lang", "video_digest"],
        )
    )

    humman_message_prompt = HumanMessagePromptTemplate.from_template('{input}')
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, humman_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     verbose=verbose)

    return chain

digest_teaching_chain = get_digest_teaching_chain()


from langchain.chains import SequentialChain

digest_teaching_overall_chain = SequentialChain(
    chains=[video_digest_chain, digest_teaching_chain],
    input_variables=["learning_lang", "user_lang", "video_content", "input"],
    # Here we return multiple variables
    output_variables=["video_digest", "text"],
    verbose=verbose)


######################################
# 單字學習提示設計
######################################
# 將文件轉換為帶行數 以及 原始內容的資料結構， 例如 { "line_number": 1, "text": "這是第一行" }
def convert_to_line_data(document_lines):
    line_data = []
    for line_number, line in enumerate(document_lines):
        line_data.append({"line_no": line_number, "text": line})
    return line_data


def random_sentence_selection_func(inputs: dict) -> dict:
    video_content = inputs["video_content"]
    line_data = convert_to_line_data(video_content)

    # 然後隨機挑選 20 句例句來模擬挑選出來的教學例句，
    import random
    random.seed(0)
    recommend_lines = random.sample(line_data, 3)

    # 依照 Line_no 來排序
    recommend_lines = sorted(recommend_lines, key=lambda x: x["line_no"])

    return {
        "selected_captions": recommend_lines
    }

sentence_random_selection_chain = TransformChain(
    input_variables=[],
    output_variables=["selected_captions"],
    transform=random_sentence_selection_func
)


output_parser_lex_learning = CommaSeparatedListOutputParser()

def get_lex_selection_chain():
    template="""你是一個專業的外語老師，你有一個特殊專長是在能夠以影片的內容找出值得教學的內容，你也很擅長做課程的規劃。

    接下來我會提供給挑選出來的教學例句，請在所有教學例句裏面隨機找出 5 個左右值得拿來做教學的詞彙。

    教學例句: {selected_captions}

    {format_instructions}
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["selected_captions"],
            partial_variables={"format_instructions": output_parser_lex_learning.get_format_instructions()}
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     output_key='selected_lexicons',
                     verbose=verbose)

    return chain

lex_selection_chain = get_lex_selection_chain()


# lex teaching chain design
def get_lex_teaching_chain():
    template="""你是一個專業的語言教材編輯，我會提供給你幾個要教學的詞彙，請你以親切且面對面的口吻跟對方教授那幾個詞彙。
    如果你的教學中有提到例句的話，也請使用 {learning_lang} 來舉例。

    教學詞彙: {selected_lexicons}
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["learning_lang", "selected_lexicons"],
        )
    )

    humman_message_prompt = HumanMessagePromptTemplate.from_template('{input}')
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, humman_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     verbose=verbose)

    return chain

lex_teaching_chain = get_lex_teaching_chain()


from langchain.chains import SequentialChain

lex_teaching_overall_chain = SequentialChain(
    chains=[sentence_random_selection_chain, lex_selection_chain, lex_teaching_chain],
    input_variables=["learning_lang", "video_content", "input"],
    # Here we return multiple variables
    output_variables=["selected_captions", "selected_lexicons", "text"],
    verbose=verbose)



######################################
# 精彩例句的文法解析提示設計
######################################
def get_caption_random_pick_chain():
    template="""你是一個專業的外語老師，你有一個特殊專長是在能夠以影片的內容找出值得教學的內容，你也很擅長做課程的規劃。

    接下來我會提供一段影片內容，請在裏面隨機找出 1 個拿來做教學的例句。

    影片內容: {video_content}

    你只要回應挑選的例句，不要其他的訊息。我要將你的 輸出做 raw data 使用。
    """
    # {format_instructions}
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["video_content"],
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     output_key='random_pick_caption',
                     #output_parser=output_parser_selected_caption,
                     verbose=verbose)

    return chain

caption_random_pick_chain = get_caption_random_pick_chain()


def get_caption_translate_chain():
    template="""你是一個專業的翻譯，你的工作是將下面例句做翻譯：

    翻譯語言： {user_lang}
    例句: {random_pick_caption}

    你只要回應翻譯後的例句，不要其他的訊息。我要將你的 輸出做 raw data 使用。
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["user_lang", "random_pick_caption"],
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     output_key='translated_caption',
                     verbose=verbose)

    return chain

translate_caption_chain = get_caption_translate_chain()


def get_gramma_intro_chain():
    template="""你是一個教材編輯，我會提供給你一個教學例句，請你針對它裏面用到的文法以及單字做介紹。

    教學例句： {random_pick_caption}
    介紹用的語言: {user_lang}

    你只要回應介紹的內容，不要其他的訊息。我要將你的 輸出做 raw data 使用。
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["user_lang", "random_pick_caption"],
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     output_key='gramma_intro',
                     verbose=verbose)

    return chain

gramma_intro_chain = get_gramma_intro_chain()


def get_gramma_teaching_chain():
    template="""你是一個專業的語言學習總編輯。請你根據下方背景資料以及要求格式回覆使用者。

    背景資料
    說明使用的語言: {user_lang}
    學生學習中的語言： {learning_lang}
    教學例句： {random_pick_caption}
    例句的翻譯： {translated_caption}
    語法介紹： {gramma_intro}
    使用者的訊息： {input}

    要求格式
    <對使用者訊息的簡單回覆，並且說明請參考下方學習資料>

    教學例句： <這裏是教學例句>
    <教學例句的翻譯>

    文法介紹： <這裏是文法介紹》
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["user_lang", "learning_lang", "random_pick_caption", "translated_caption", "gramma_intro", "input"],
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     verbose=verbose)

    return chain

gramma_teaching_chain = get_gramma_teaching_chain()


gramma_teaching_overall_chain = SequentialChain(
    chains=[caption_random_pick_chain, translate_caption_chain, gramma_intro_chain, gramma_teaching_chain],
    input_variables=["user_lang", "learning_lang", "video_content", "input"],
    # Here we return multiple variables
    output_variables=["random_pick_caption", "translated_caption", "gramma_intro", "text"],
    #output_variables=["video_source"],
    verbose=verbose)


######################################
# 延伸學習提示設計
######################################
def get_related_lex_recommend_chain():
    template="""你是一個專業的語言學家以及雙語外語老師。
    請你針對學生訊息中提出的詞彙，推薦幾個相關的學習詞彙以及例句。

    你推薦的詞彙以及例句也請包含使用者學習中的語言 以及 使用者的母語。

    使用者學習中的語言： {learning_lang}
    使用者的母語： {user_lang}
    使用者的訊息： {input}

    同時也請你以下方格式回覆

    <使用者訊息的簡單回覆>
    <延伸學習的一些訊息在這裏，這裏的格式讓你自由發揮，看起來通順最重要>
    <引導使用者做下個動作>
    """
    system_message_prompt = SystemMessagePromptTemplate(
        prompt = PromptTemplate(
            template = template,
            input_variables=["user_lang", "learning_lang", "input"],
        )
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chain = LLMChain(llm=llm_chat,
                     prompt=chat_prompt,
                     verbose=verbose)

    return chain

related_lex_chain = get_related_lex_recommend_chain()


def get_learning_mode_guide_chain():
    template = """我們是一個透過 youtube影片內容來學習語言的線上學習系統，我們已經透過使用者的訊息選擇了接下來的學習影片，請你接下來引導使用者選擇他想要的學習方式。

    背景資料：
    使用者的母語： {user_lang}
    使用者想學習的語言： {learning_lang}
    使用者選擇的學習影片： {video_source}

    我們提供的學習方式有：
    1. 影片內容摘要： 讓使用者可以快速了解影片內容
    2. 詞彙學習： 使用影片內使用過的詞彙來學習
    3. 文法學習： 使用影片內的例句來做文法學習
    4. 延伸詞彙學習： 任何一個使用者提出的詞彙的相關詞介紹
    """

    system_message_prompt_default = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template = template,
                input_variables=["user_lang", "learning_lang", "video_source"],
            )
        )

    chat_prompt_template = ChatPromptTemplate.from_messages([
        system_message_prompt_default,
        #MessagesPlaceholder(variable_name="history"),  # key in memory
        HumanMessagePromptTemplate.from_template('{input}')
    ])

    chain = LLMChain(llm=llm_chat, prompt=chat_prompt_template, memory=memory, verbose=verbose)

    return chain

learning_mode_guide_chain = get_learning_mode_guide_chain()


######################################
# 學習模式的 router chain 設計
######################################
def get_learning_mode_router_chain():
    # setup for LLMRouterChain
    prompt_infos = [
        {
            "name": "DigestLearning",
            "description": "需要快速瞭解影片內容時",
            "chain": digest_teaching_overall_chain
        },
        {
            "name": "LexLearning",
            "description": "專長處理影片內詞彙的學習",
            "chain": lex_teaching_overall_chain,
        },
        {
            "name": "CaptionLearning",
            "description": "專長處理文法以及透過例句來學習",
            "chain": gramma_teaching_overall_chain
        },
        {
            "name": "RelatedLexLearning",
            "description": "專長處理延伸詞彙的學習",
            "chain": related_lex_chain
        },
    ]

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm_chat, router_prompt)


    return router_chain

learning_mode_router_chain = get_learning_mode_router_chain()


######################################
# 學習模式下的 routering handling
######################################
def leanring_mode_phase_handler(user_lang, learning_lang, user_input: str) -> str:
    global chatting_mode

    # chat mode handler
    router_result = learning_mode_router_chain(user_input)

    if verbose:
        print(f'router_result: {router_result}')

    ai_response = None
    cur_destination = router_result['destination']
    if cur_destination== 'DigestLearning':
        ai_response = digest_teaching_overall_chain({
            'user_lang': user_lang,
            'learning_lang': learning_lang,
            'input': user_input,
            'video_content': learning_document,
        })
    elif cur_destination == 'LexLearning':
        ai_response = lex_teaching_overall_chain({
            'learning_lang': learning_lang,
            'video_content': learning_document,
            'input': user_input
        })
    elif cur_destination == 'CaptionLearning':
        ai_response = gramma_teaching_overall_chain({
            'user_lang': user_lang,
            'learning_lang': learning_lang,
            'video_content': learning_document,
            'input': user_input,
        })
    elif cur_destination == 'RelatedLexLearning':
        ai_response = related_lex_chain({
            'user_lang': user_lang,
            'learning_lang': learning_lang,
            'input': user_input,
        })
    else:
        ai_response = learning_mode_guide_chain({
            'user_lang': user_lang,
            'learning_lang': learning_lang,
            'video_source': default_youtube_video_file,
            'input': user_input
        })

    return ai_response["text"]