from aiohttp import web

# necessary to load bot
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from src.bot import Bot, Context, load_w2v

async def home(request: web.Request) -> web.Response:
    return web.FileResponse('../frontend/build/index.html')

async def bot_endpoint(request: web.Request) -> web.Response:
    data = await request.json()
    bot = request.app['bot']
    context = Context(data['context'])
    message, context = bot.get_response(data['message'], context)
    return web.json_response({
        'message': message,
        'context': context
    })

@web.middleware
async def cors_middleware(request, handler):
    # only necessary for running frontend in development mode
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:3000'
    response.headers['Access-Control-Allow-Methods'] = '*'
    return response


if __name__=='__main__':
    middlewares = [cors_middleware]
    app = web.Application(middlewares=middlewares)
    model = load_w2v()
    app['bot'] = Bot('src/data.yaml', model=model)
    app.add_routes([
        web.post('/api', bot_endpoint),
        web.get('/', home),
        web.static('/static/', '../frontend/build/static/'),
    ])
    web.run_app(app)
