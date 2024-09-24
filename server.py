import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
# Initialize sentiment analysis tools
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews from CSV
reviews = pd.read_csv('data/reviews.csv').to_dict('records')


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """
        # Analyze sentiment for each review
        for review in reviews:
            review_body = review.get("ReviewBody")
            sentiment_scores = self.analyze_sentiment(review_body)
            review["sentiment"] = sentiment_scores

        if environ["REQUEST_METHOD"] == "GET":
            # Extract the query string
            query_string = environ.get("QUERY_STRING", "")
            params = parse_qs(query_string)

            # Extract parameters with default values if not provided
            location = params.get("location", [None])[0]
            start_date = params.get("start_date", [None])[0]
            end_date = params.get("end_date", [None])[0]

            # Filter reviews based on parameters
            filtered_reviews = reviews
            if location:
                filtered_reviews = [
                    review for review in filtered_reviews if review.get("Location") == location
                ]
            if start_date and end_date:
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(start_date, "%Y-%m-%d") <= datetime.strptime(review.get("Timestamp"), "%Y-%m-%d %H:%M:%S") <= datetime.strptime(end_date, "%Y-%m-%d")
                ]
            elif start_date:
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(start_date, "%Y-%m-%d") <= datetime.strptime(review.get("Timestamp"), "%Y-%m-%d %H:%M:%S")
                ]
            elif end_date:
                filtered_reviews = [
                    review for review in filtered_reviews
                    if datetime.strptime(review.get("Timestamp"), "%Y-%m-%d %H:%M:%S") <= datetime.strptime(end_date, "%Y-%m-%d")
                ]

            # Create the response body from the filtered reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            # Extract the request body
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(content_length).decode("utf-8")
            params = parse_qs(request_body)

            # Extract parameters with default values if not provided
            review_body = params.get("ReviewBody", [None])[0]
            location = params.get("Location", [None])[0]

            if review_body is None or location is None:
                response_body = "Missing input. Please provide both ReviewBody and Location."
                response_body = response_body.encode("utf-8")

                # Set the response headers
                start_response("400 Unauthorized", [
                    ("Content-Type", "text/plain"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            # Validate location
            valid_locations = set(review.get("Location") for review in reviews)
            if location not in valid_locations:
                response_body = "Invalid location. Please provide a valid location."
                response_body = response_body.encode("utf-8")

                # Set the response headers
                start_response("400 Bad Request", [
                    ("Content-Type", "text/plain"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            # Add Timestamp and ReviewId to the review
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            review_id = str(uuid.uuid4())

            # Create a new review dictionary
            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp
            }

            # Add the new review to the reviews list
            reviews.append(new_review)

            # Set the response body
            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            # Set the response headers
            start_response("201 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()