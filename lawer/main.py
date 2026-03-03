from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config 
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import config


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


class RagService(object):
    def __init__(self):       
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一位精通中国法律的专业人士。请依据以下检索到的法律条例，准确、专业地回答用户的问题。回答时应优先使用文档中的信息，并结合你的法律知识，但必须确保所有陈述都有据可依，尽可能引用具体的法律条文或来源。如果文档内容不足以回答问题，请明确告知“根据提供的材料无法回答该问题”，切勿编造信息。回答需清晰，符合法律专业表达习惯。"                
                 "检索到的法律条例如下:{context}。"),               
                ("user", "{input}")
            ]
        )

        self.chat_model = ChatOllama(model="lawer")
        self.embeddings = OllamaEmbeddings(model=config.embeddings_model)
        self.chain = self.__get_chain()

    def __get_chain(self):
        vectorstore = Chroma(
        persist_directory=config.persist_dir,
        embedding_function=self.embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"参考文档：{doc.page_content}\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | self.prompt_template |  self.chat_model | StrOutputParser()
        )

        return chain


if __name__ == "__main__":
    
    res = RagService().chain.invoke({"input": "我和别人签了租赁合同，对方现在不付钱了，我还能不能要回这笔钱？"})
    print(res)