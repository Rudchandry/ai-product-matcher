from main import app

# Vercel entry point
def handler(request):
    return app(request)

# Alternative handler
application = app