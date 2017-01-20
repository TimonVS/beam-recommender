# Beam Recommender

## Development

1. [Install Docker (with Docker Compose)](https://docs.docker.com/compose/install/)
2. Clone this repository
3. Run `docker-compose build .` to build the container
4. Run `docker-compose up` to start the container

## Configuration

_If you haven't created a Reddit application yet, create a new application on <https://www.reddit.com/prefs/apps>._

Create a `.env` file in the root of the project and fill in the following details from your Reddit application:

```
REDDIT_CLIENT_ID=<reddit client id here>
REDDIT_CLIENT_SECRET=<reddit client secret here>
REDDIT_USER_AGENT=<descriptive user agent here>
```
