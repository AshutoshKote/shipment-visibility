.PHONY: help up down logs ps topic simulate dry verify bronze query ui clean

help:
	@echo "Targets:"
	@echo "  up        - start Kafka + Kafka UI (docker)"
	@echo "  down      - stop containers"
	@echo "  topic     - create the gps.pings topic (6 partitions)"
	@echo "  simulate  - run the GPS simulator -> Kafka"
	@echo "  dry       - run the simulator in dry-run (print, no Kafka)"
	@echo "  verify    - consume a few pings from Kafka (sanity check)"
	@echo "  bronze    - run Spark Structured Streaming -> Bronze Delta"
	@echo "  silver    - Bronze -> Silver (clean, dedup, watermark, enrich)"
	@echo "  gold      - Silver -> Gold (status events + windowed exceptions)"
	@echo "  query     - batch-read the Bronze Delta table"
	@echo "  query-gold- batch-read the Silver + Gold tables"
	@echo "  gen-data  - generate the ETA training dataset"
	@echo "  train     - train the ETA model + log to MLflow"
	@echo "  mlflow-ui - open the MLflow UI (port 5050)"
	@echo "  serve     - run the FastAPI ETA service (port 8000)"
	@echo "  predict   - send a batch of predictions to the API"
	@echo "  predict-shift - send drifted predictions (to demo drift)"
	@echo "  drift     - run the PSI drift monitor"
	@echo "  retrain   - champion/challenger retrain + conditional promote"
	@echo "  dashboard - run the live Streamlit dashboard (port 8501)"
	@echo "  ui        - print the Kafka UI URL"
	@echo "  clean     - delete local data/ (Delta + checkpoints)"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f kafka

ps:
	docker compose ps

topic:
	docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists \
		--topic gps.pings --partitions 6 --replication-factor 1 \
		--bootstrap-server localhost:9092
	docker exec kafka /opt/kafka/bin/kafka-topics.sh --describe \
		--topic gps.pings --bootstrap-server localhost:9092

simulate:
	python simulator/gps_simulator.py --shipments 20 --interval 2 --late-prob 0.1

dry:
	python simulator/gps_simulator.py --shipments 5 --interval 1 --duration 5 --dry-run

verify:
	python ingestion/verify_consumer.py

bronze:
	python ingestion/bronze_ingest.py

silver:
	python processing/silver_clean.py

gold:
	python processing/gold_exceptions.py

query:
	python ingestion/query_bronze.py

query-gold:
	python processing/query_gold.py

gen-data:
	python ml/generate_training_data.py --trips 1200

train:
	python ml/train_eta.py

mlflow-ui:
	@echo "MLflow UI -> http://localhost:5050"
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050

serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000

predict:
	python serving/load_test.py --n 250

predict-shift:
	python serving/load_test.py --n 250 --shift

drift:
	python monitoring/drift_monitor.py

retrain:
	python ml/retrain.py

dashboard:
	streamlit run dashboard/app.py

ui:
	@echo "Kafka UI -> http://localhost:8080"

clean:
	rm -rf data
