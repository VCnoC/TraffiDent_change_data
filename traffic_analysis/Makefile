# Makefile for Traffic Analysis Project

.PHONY: help install test test-cov lint clean run run-test run-full run-classify

help:
	@echo "交通分析项目 - 可用命令："
	@echo ""
	@echo "  make install      - 安装项目依赖"
	@echo "  make test         - 运行测试"
	@echo "  make test-cov     - 运行测试并生成覆盖率报告"
	@echo "  make lint         - 代码风格检查"
	@echo "  make clean        - 清理临时文件"
	@echo "  make run          - 运行程序（处理前100个事故）"
	@echo "  make run-test     - 测试模式（仅10个事故）"
	@echo "  make run-full     - 完整处理所有事故"
	@echo "  make run-classify - 分类处理所有事故"
	@echo ""

install:
	pip install -r requirements.txt

test:
	pytest -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term
	@echo "覆盖率报告已生成：htmlcov/index.html"

lint:
	@echo "运行 flake8..."
	flake8 . --max-line-length=100 --exclude=venv,__pycache__,.pytest_cache
	@echo "代码检查完成！"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage
	@echo "清理完成！"

run:
	python main.py --data-dir ../data --output-dir ./output

run-test:
	python main.py --test

run-full:
	python main.py --data-dir ../data --output-dir ./output --full

run-classify:
	python main.py --data-dir ../data --output-dir ./output --classify --full
