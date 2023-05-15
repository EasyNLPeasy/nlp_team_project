# NLP Project: acquire.py

"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import itertools
import time

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['./WebGoat/WebGoat', './vercel/next-react-server-components', './LaurentMazare/tch-rs', './ultralytics/yolov5', './FastForwardTeam/FastForward', './neuml/txtai', './microsoft/Web-Dev-For-Beginners', './postgresml/postgresml', './ajeetdsouza/zoxide', './pyg-team/pytorch_geometric', './GoogleChrome/lighthouse', './facebook/metro', './facebookresearch/demucs', './TapiocaFox/Daijishou', './rustdesk/rustdesk', './scraly/developers-conferences-agenda', './paradigmxyz/reth', './benphelps/homepage', './pynecone-io/pynecone', './pmndrs/drei', './Yoast/wordpress-seo', './paritytech/substrate', './goldbergyoni/javascript-testing-best-practices', './airbnb/javascript', './libreddit/libreddit', './GyulyVGC/sniffnet', './yjs/yjs', './THUDM/CodeGeeX', './quickwit-oss/quickwit', './riffusion/riffusion', './poteto/hiring-without-whiteboards', './rapiz1/rathole', './meilisearch/meilisearch', './catppuccin/gtk', './acantril/learn-cantrill-io-labs', './blocklistproject/Lists', './MrXujiang/h5-Dooring', './DioxusLabs/dioxus', './lucidrains/PaLM-rlhf-pytorch', './donnemartin/system-design-primer', './gfx-rs/wgpu', './hrkfdn/ncspot', './emilk/egui', './axios/axios', './jaywcjlove/awesome-mac', './EleutherAI/lm-evaluation-harness', './vitejs/awesome-vite', './nasa/openmct', './EddieHubCommunity/LinkFree', './grocy/grocy', './Expensify/App', './tokio-rs/tokio', './BlinkDL/RWKV-LM', './Sanster/lama-cleaner', './naptha/tesseract.js', './nushell/nushell', './Yamato-Security/hayabusa', './trekhleb/javascript-algorithms', './cberner/redb', './StevenBlack/hosts', './wekan/wekan', './apache/airflow', './d8ahazard/sd_dreambooth_extension', './KaTeX/KaTeX', './SBoudrias/Inquirer.js', './burn-rs/burn', './paritytech/polkadot', './EmulatorJS/EmulatorJS', './iced-rs/iced', './ashawkey/stable-dreamfusion', './young-geng/EasyLM', './cozodb/cozo', './TheAlgorithms/Rust', './3kh0/website-v3', './tikv/tikv', './sarah-ek/faer-rs', './MetaMask/eth-phishing-detect', './facebookresearch/encodec', './HazyResearch/flash-attention', './mosaicml/composer', './NVIDIA/Megatron-LM', './freeCodeCamp/boilerplate-npm', './0x192/universal-android-debloater', './GuillaumeGomez/sysinfo', './elementor/elementor', './tinyvision/SOLIDER', './bmaltais/kohya_ss', './microsoft/DeepSpeed', './a16z/helios', './sdatkinson/NeuralAmpModelerPlugin', './yt-dlp/yt-dlp', './datafuselabs/databend', './neondatabase/neon', './typicode/json-server', './toriato/stable-diffusion-webui-wd14-tagger', './Tencent/cherry-markdown', './chroma-core/chroma', './A-d-i-t-h-y-a-n/hermit-md', './slint-ui/slint', './openai/whisper', './LGUG2Z/komorebi', './qdrant/qdrant', './marticliment/WingetUI', './mailcow/mailcow-dockerized', './PawanOsman/ChatGPT', './TheAlgorithms/JavaScript', './OpenBB-finance/OpenBBTerminal', './farshadz1997/Microsoft-Rewards-bot', './rust-unofficial/awesome-rust', './Horus645/swww', './neetcode-gh/leetcode', './raspberrypi/usbboot', './AUTOMATIC1111/stable-diffusion-webui', './RasaHQ/rasa', './volta-cli/volta', './benf2004/ChatGPT-Prompt-Genius', './processing/p5.js', './appium/appium', './maplibre/martin', './walkxcode/dashboard-icons', './Dong-learn9/TVBox-zyjk', './InstaPy/InstaPy', './NidukaAkalanka/x-ui-english', './rawandahmad698/PyChatGPT', './chidiwilliams/buzz', './casey/ord', './C-Nedelcu/talk-to-chatgpt', './martinvonz/jj', './Zero6992/chatGPT-discord-bot', './cmdr2/stable-diffusion-ui', './DataDog/dd-trace-js', './charliermarsh/ruff', './enso-org/enso', './umutxyp/MusicBot', './huggingface/transformers', './coqui-ai/TTS', './OCA/web', './leptos-rs/leptos', './LAION-AI/Open-Assistant', './RadeonOpenCompute/ROCm', './Byron/gitoxide', './nolimits4web/swiper', './helix-editor/helix', './windmill-labs/windmill', './AleoHQ/snarkOS', './Z4nzu/hackingtool', './neonbjb/tortoise-tts', './ClementTsang/bottom', './MetaMask/metamask-extension', './OpenTalker/video-retalking', './pittcsc/Summer2023-Internships', './an-anime-team/an-anime-game-launcher', './mishoo/UglifyJS', './sudheerj/reactjs-interview-questions', './hwchase17/langchain', './pola-rs/polars', './huggingface/text-generation-inference', './fermyon/spin', './dortania/OpenCore-Legacy-Patcher', './deepinsight/insightface', './DominikDoom/a1111-sd-webui-tagcomplete', './zloirock/core-js', './ChanseyIsTheBest/NX-60FPS-RES-GFX-Cheats', './elebumm/RedditVideoMakerBot']

 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        # time.sleep(13)
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        # time.sleep(11)
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme."):
            # time.sleep(10)
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)