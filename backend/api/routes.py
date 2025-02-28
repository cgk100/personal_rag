@router.post("/api/test-connection")
async def test_connection(request: Request):
    """
    测试与LLM API的连接
    """
    try:
        # 解析请求体
        data = await request.json()
        base_url = data.get("baseUrl")
        api_key = data.get("apiKey")
        model = data.get("model")
        
        # 记录请求
        logger.info(f"收到测试连接请求: {model}")
        
        # 初始化LLM客户端
        from backend.llm_client import LLMClient
        llm_client = LLMClient(
            base_url=base_url,
            api_key=api_key,
            model=model
        )
        
        # 测试连接
        start_time = time.time()
        response = llm_client.test_connection()
        end_time = time.time()
        
        # 计算响应时间
        response_time = int((end_time - start_time) * 1000)
        
        return {
            "success": True,
            "model": model,
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"测试连接失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/api/files")
async def get_files():
    """
    获取知识库文件列表
    """
    try:
        # 获取文件列表
        files = []
        
        # 遍历知识库目录
        knowledge_dir = os.path.join(os.getcwd(), "knowledge")
        if os.path.exists(knowledge_dir):
            for filename in os.listdir(knowledge_dir):
                file_path = os.path.join(knowledge_dir, filename)
                if os.path.isfile(file_path):
                    # 获取文件信息
                    file_info = {
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "last_modified": os.path.getmtime(file_path)
                    }
                    files.append(file_info)
        
        return {"files": files}
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}") 