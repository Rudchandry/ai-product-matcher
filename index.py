def handler(event, context):
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': '{"message": "AI Product Matcher - Root Level", "status": "working", "location": "root_index"}'
    }