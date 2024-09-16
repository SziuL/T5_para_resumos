from typing import LiteralString
from bs4 import BeautifulSoup
import requests
import chardet


def obter_conteudo_pagina(url):
    """
    Função que faz a requisição HTTP para o URL fornecido e retorna o conteúdo da página
    caso a resposta seja válida.
    """
    retorno = requests.get(url)
    if retorno.status_code == 200:
        return retorno.text
    else:
        raise Exception("Falha em obter o conteúdo!")


def detectar_encoding(conteudo):
    """
    Função que detecta a codificação do conteúdo HTML.
    """
    encoding = chardet.detect(conteudo)
    return encoding["encoding"]


def parse_conteudo_html(conteudo_html):
    """
    Função que analisa o conteúdo HTML e retorna um objeto BeautifulSoup.
    """
    return BeautifulSoup(conteudo_html, "html.parser")


def extrair_texto_relevante(soup) -> LiteralString:
    """
    Função que extrai o texto relevante (parágrafos) de uma página HTML analisada.
    """
    paragrafos = soup.find_all("p")
    texto = " ".join(
        [para.get_text() for para in paragrafos if para.get_text().strip()]
    )
    return texto
