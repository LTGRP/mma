import requests
import urllib3
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
pd.set_option('display.max_columns', 500)

from .scripts import get_fight_stats, get_fight_card, get_all_fight_stats, get_fighter_details, update_fight_stats, update_fighter_details