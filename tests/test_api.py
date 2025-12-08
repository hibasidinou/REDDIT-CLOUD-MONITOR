import praw
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

reddit = praw.Reddit(
    client_id=config['reddit']['client_id'],
    client_secret=config['reddit']['client_secret'],
    user_agent=config['reddit']['user_agent']
)

print(f"✅ Connecté en tant que: {reddit.user.me()}")