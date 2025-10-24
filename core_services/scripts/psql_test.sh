#!/bin/bash

# PostgreSQL Connection & Test Script via Docker
# ===============================================
# This script provides easy access to the PostgreSQL database running in Docker
# and includes built-in test queries for validation.
#
# Usage Examples:
#   ./scripts/psql_test.sh                       # Interactive mode
#   ./scripts/psql_test.sh -c "SELECT * FROM stocks LIMIT 5;"  # Execute query
#
# Test Commands:
#   ./scripts/psql_test.sh test                  # Run all tests
#   ./scripts/psql_test.sh test-stocks           # Test stocks query
#   ./scripts/psql_test.sh test-prices           # Test latest prices
#   ./scripts/psql_test.sh test-stock VNM        # Test specific stock
#   ./scripts/psql_test.sh test-indices          # Test market indices
#   ./scripts/psql_test.sh test-count            # Count all records
#
# Useful Queries:
#
# Check stocks:
#   SELECT * FROM stocks LIMIT 5;
#
# Check latest prices:
#   SELECT s.symbol, p.date, p.close, p.volume
#   FROM stocks s
#   JOIN stock_prices p ON s.id = p.stock_id
#   ORDER BY p.date DESC
#   LIMIT 10;
#
# Get prices for specific stock:
#   SELECT s.symbol, p.date, p.open, p.high, p.low, p.close, p.volume
#   FROM stocks s
#   JOIN stock_prices p ON s.id = p.stock_id
#   WHERE s.symbol = 'VNM'
#   ORDER BY p.date DESC
#   LIMIT 10;
#
# Check market indices:
#   SELECT * FROM indices;
#
# Count records:
#   SELECT 
#     (SELECT COUNT(*) FROM stocks) as total_stocks,
#     (SELECT COUNT(*) FROM stock_prices) as total_prices,
#     (SELECT COUNT(*) FROM indices) as total_indices;
#
# Connection URL (for reference - use docker exec instead):
#   postgresql://user:password@localhost:5432/stockaids

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="stockaids-backend-postgres-1"
DB_USER="user"
DB_NAME="stockaids"

# Check if Docker container is running
check_container() {
    if ! docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
        echo -e "${RED}Error: Docker container '${CONTAINER_NAME}' is not running${NC}"
        echo -e "${YELLOW}Start the container with:${NC}"
        echo -e "  cd ../stockaids-backend && docker-compose up -d postgres"
        exit 1
    fi
}

# Print connection info
print_info() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}PostgreSQL Database Connection${NC}                            ${BLUE}║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC}  Container: ${YELLOW}${CONTAINER_NAME}${NC}"
    echo -e "${BLUE}║${NC}  Database:  ${YELLOW}${DB_NAME}${NC}"
    echo -e "${BLUE}║${NC}  User:      ${YELLOW}${DB_USER}${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Print quick reference
print_quick_reference() {
    echo -e "${GREEN}Quick Reference:${NC}"
    echo -e "  ${YELLOW}\\dt${NC}              - List all tables"
    echo -e "  ${YELLOW}\\d stocks${NC}        - Describe stocks table"
    echo -e "  ${YELLOW}\\q${NC}               - Quit"
    echo -e "  ${YELLOW}\\?${NC}               - Show all psql commands"
    echo ""
    echo -e "${GREEN}Sample Queries:${NC}"
    echo -e "  ${YELLOW}SELECT * FROM stocks LIMIT 5;${NC}"
    echo -e "  ${YELLOW}SELECT COUNT(*) FROM stock_prices;${NC}"
    echo ""
    echo -e "${GREEN}Test Functions:${NC}"
    echo -e "  ${YELLOW}./scripts/psql_test.sh test${NC}              - Run all test queries"
    echo -e "  ${YELLOW}./scripts/psql_test.sh test-stocks${NC}       - Test stocks query"
    echo -e "  ${YELLOW}./scripts/psql_test.sh test-prices${NC}       - Test latest prices"
    echo -e "  ${YELLOW}./scripts/psql_test.sh test-indices${NC}      - Test market indices"
    echo ""
}

# Test functions
test_stocks() {
    echo -e "${GREEN}Testing: Check stocks${NC}"
    docker exec "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT * FROM stocks LIMIT 5;"
}

test_latest_prices() {
    echo -e "${GREEN}Testing: Check latest prices${NC}"
    docker exec "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT s.symbol, p.date, p.close, p.volume FROM stocks s JOIN stock_prices p ON s.id = p.stock_id ORDER BY p.date DESC LIMIT 10;"
}

test_specific_stock() {
    local symbol=${1:-VNM}
    echo -e "${GREEN}Testing: Get prices for stock ${YELLOW}${symbol}${NC}"
    docker exec "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT s.symbol, p.date, p.open, p.high, p.low, p.close, p.volume FROM stocks s JOIN stock_prices p ON s.id = p.stock_id WHERE s.symbol = '${symbol}' ORDER BY p.date DESC LIMIT 10;"
}

test_market_indices() {
    echo -e "${GREEN}Testing: Check market indices${NC}"
    docker exec "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT * FROM indices;"
}

test_count_records() {
    echo -e "${GREEN}Testing: Count all records${NC}"
    docker exec "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT (SELECT COUNT(*) FROM stocks) as total_stocks, (SELECT COUNT(*) FROM stock_prices) as total_prices, (SELECT COUNT(*) FROM indices) as total_indices;"
}

# Run all tests
run_all_tests() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}Running All Database Tests${NC}                                ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    test_count_records
    echo ""
    
    test_stocks
    echo ""
    
    test_latest_prices
    echo ""
    
    test_specific_stock "VNM"
    echo ""
    
    test_market_indices
    echo ""
    
    echo -e "${GREEN}✓ All tests completed!${NC}"
}

# Main execution
main() {
    check_container
    
    # Handle different commands
    case "${1}" in
        test)
            run_all_tests
            ;;
        test-stocks)
            test_stocks
            ;;
        test-prices)
            test_latest_prices
            ;;
        test-stock)
            test_specific_stock "${2}"
            ;;
        test-indices)
            test_market_indices
            ;;
        test-count)
            test_count_records
            ;;
        -c|--command)
            shift
            docker exec -it "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" "$@"
            ;;
        "")
            print_info
            print_quick_reference
            echo -e "${GREEN}Connecting to PostgreSQL...${NC}\n"
            docker exec -it "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}"
            ;;
        *)
            # Pass all arguments to psql
            docker exec -it "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" "$@"
            ;;
    esac
}

# Run main function
main "$@"
