echo "Waiting for PostgreSQL to be ready..."
while ! nc -z db 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

python manage.py migrate

# Create a superuser (optional, comment out if not needed)
# python manage.py createsuperuser --noinput

python manage.py runserver 0.0.0.0:8000
