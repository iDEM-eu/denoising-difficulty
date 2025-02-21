name: CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Tests
        run: |
          pytest tests/

      - name: Log in to GitHub Container Registry
        run: echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u $GITHUB_ACTOR --password-stdin

      - name: Build Docker Image
        run: |
          docker build -t $REGISTRY/$IMAGE_NAME:latest .

      - name: Push Docker Image to GitHub Container Registry
        run: |
          docker push $REGISTRY/$IMAGE_NAME:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: SSH into Server and Deploy
        uses: appleboy/ssh-action@v0.1.6
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            docker pull $REGISTRY/$IMAGE_NAME:latest
            docker stop text-model-api || true
            docker rm text-model-api || true
            docker run -d --name text-model-api -p 8000:8000 $REGISTRY/$IMAGE_NAME:latest

      - name: Send Slack Notification
        uses: rtCamp/action-slack-notify@v2
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_CHANNEL: deployments
          SLACK_MESSAGE: "Deployment successful for ${{ github.repository }}"

      - name: Send Email Notification
        uses: dawidd6/action-send-mail@v3
        with:
          server_address: ${{ secrets.MAIL_SERVER }}
          server_port: ${{ secrets.MAIL_PORT }}
          username: ${{ secrets.MAIL_USERNAME }}
          password: ${{ secrets.MAIL_PASSWORD }}
          subject: "Deployment Success: ${{ github.repository }}"
          to: ${{ secrets.NOTIFICATION_EMAIL }}
          from: "GitHub Actions <no-reply@github.com>"
          body: "The deployment for ${{ github.repository }} was successful."
