#!/bin/sh
set -e

echo "Waiting for PostgreSQL..."
until pg_isready -h "${PG_HOST:-db}" -p "${PG_PORT:-5432}" -U "${PG_USER:-postgres}"; do
  sleep 1
done

echo "Running migrations..."
python manage.py migrate --noinput

echo "Starting gunicorn..."
exec gunicorn config.wsgi:application \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers 2 \
  --timeout 120 \
  --access-logfile -
