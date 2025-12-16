import json
from pathlib import Path

from discogs_client.exceptions import HTTPError
from django.conf import settings 
from django.core.management.base import BaseCommand