#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bayes_emc {

struct Observation {
    std::vector<double> x;
    std::vector<double> y;
};

enum class DataFormat {
    Whitespace,
    Csv,
    Tsv,
};

struct DataTableOptions {
    DataFormat format = DataFormat::Whitespace;
    bool header = false;
    std::vector<std::string> input_columns;
    std::vector<std::string> output_columns;
};

class DataSet {
    public:
        DataSet() = default;

        DataSet(
            const std::size_t input_dim,
            const std::size_t output_dim,
            std::vector<Observation> observations
        ) : input_dim_(input_dim), output_dim_(output_dim), observations_(std::move(observations)) {
            Validate();
        }

        static DataSet LoadWhitespace(
            const std::string & path,
            const std::size_t input_dim,
            const std::size_t output_dim
        ) {
            DataTableOptions options;
            options.format = DataFormat::Whitespace;
            return LoadTable(path, input_dim, output_dim, options);
        }

        static DataSet LoadTable(
            const std::string & path,
            const std::size_t input_dim,
            const std::size_t output_dim,
            const DataTableOptions & options = DataTableOptions{}
        ) {
            std::ifstream stream(path);
            if (!stream) {
                throw std::runtime_error("Could not open data file: " + path);
            }

            std::vector<Observation> observations;
            std::vector<std::string> header_columns;
            std::vector<std::size_t> input_indices;
            std::vector<std::size_t> output_indices;
            std::string line;
            std::size_t line_number = 0;
            while (std::getline(stream, line)) {
                ++line_number;
                const std::string stripped = Trim(line);
                if (stripped.empty() || stripped[0] == '#') continue;

                const std::vector<std::string> columns = SplitLine(stripped, options.format);
                if (options.header && header_columns.empty()) {
                    header_columns = columns;
                    continue;
                }
                if (!options.header
                    && (!options.input_columns.empty() || !options.output_columns.empty())) {
                    throw std::runtime_error("Named data columns require header=true.");
                }
                const bool named_columns = !options.input_columns.empty() || !options.output_columns.empty();
                if (input_indices.empty() && output_indices.empty()) {
                    input_indices = ResolveColumnIndices(
                        header_columns,
                        options.input_columns,
                        input_dim,
                        0,
                        "input"
                    );
                    output_indices = ResolveColumnIndices(
                        header_columns,
                        options.output_columns,
                        output_dim,
                        input_dim,
                        "output"
                    );
                }
                if (!named_columns && columns.size() != input_dim + output_dim) {
                    throw std::runtime_error(
                        "Line " + std::to_string(line_number)
                        + " has " + std::to_string(columns.size())
                        + " column(s); expected " + std::to_string(input_dim + output_dim)
                    );
                }

                Observation observation;
                observation.x.resize(input_dim);
                observation.y.resize(output_dim);
                for (std::size_t input_id = 0; input_id < input_dim; ++input_id) {
                    observation.x[input_id] = ParseColumn(columns, input_indices[input_id], line_number);
                }
                for (std::size_t output_id = 0; output_id < output_dim; ++output_id) {
                    observation.y[output_id] = ParseColumn(columns, output_indices[output_id], line_number);
                }
                observations.push_back(std::move(observation));
            }
            return DataSet(input_dim, output_dim, std::move(observations));
        }

        static DataSet FromRows(
            const std::size_t input_dim,
            const std::size_t output_dim,
            const std::vector<std::vector<double>> & rows
        ) {
            std::vector<Observation> observations;
            for (const std::vector<double> & row : rows) {
                if (row.size() != input_dim + output_dim) {
                    throw std::invalid_argument("Data row has an unexpected number of columns.");
                }
                Observation observation;
                observation.x.assign(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(input_dim));
                observation.y.assign(row.begin() + static_cast<std::ptrdiff_t>(input_dim), row.end());
                observations.push_back(std::move(observation));
            }
            return DataSet(input_dim, output_dim, std::move(observations));
        }

        std::size_t InputDim() const { return input_dim_; }
        std::size_t OutputDim() const { return output_dim_; }
        std::size_t Size() const { return observations_.size(); }
        const std::vector<Observation> & Observations() const { return observations_; }

    private:
        static std::string Trim(const std::string & value) {
            const auto begin = std::find_if_not(value.begin(), value.end(), [](const unsigned char ch) {
                return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
            });
            const auto end = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char ch) {
                return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
            }).base();
            if (begin >= end) return "";
            return std::string(begin, end);
        }

        static std::vector<std::string> SplitLine(const std::string & line, const DataFormat format) {
            if (format == DataFormat::Whitespace) {
                std::istringstream stream(line);
                std::vector<std::string> columns;
                std::string value;
                while (stream >> value) {
                    columns.push_back(value);
                }
                return columns;
            }
            const char delimiter = format == DataFormat::Csv ? ',' : '\t';
            std::vector<std::string> columns;
            std::string value;
            bool in_quotes = false;
            for (std::size_t index = 0; index < line.size(); ++index) {
                const char ch = line[index];
                if (ch == '"') {
                    if (in_quotes && index + 1 < line.size() && line[index + 1] == '"') {
                        value.push_back('"');
                        ++index;
                    } else {
                        in_quotes = !in_quotes;
                    }
                } else if (ch == delimiter && !in_quotes) {
                    columns.push_back(Trim(value));
                    value.clear();
                } else {
                    value.push_back(ch);
                }
            }
            columns.push_back(Trim(value));
            return columns;
        }

        static std::vector<std::size_t> ResolveColumnIndices(
            const std::vector<std::string> & header,
            const std::vector<std::string> & selected,
            const std::size_t expected_count,
            const std::size_t positional_offset,
            const std::string & label
        ) {
            std::vector<std::size_t> indices;
            indices.reserve(expected_count);
            if (selected.empty()) {
                for (std::size_t index = 0; index < expected_count; ++index) {
                    indices.push_back(positional_offset + index);
                }
                return indices;
            }
            if (selected.size() != expected_count) {
                throw std::runtime_error(label + "_columns has an unexpected length.");
            }
            if (header.empty()) {
                throw std::runtime_error(label + "_columns requires a header row.");
            }
            for (const std::string & name : selected) {
                const auto found = std::find(header.begin(), header.end(), name);
                if (found == header.end()) {
                    throw std::runtime_error("Data column not found: " + name);
                }
                indices.push_back(static_cast<std::size_t>(found - header.begin()));
            }
            return indices;
        }

        static double ParseColumn(
            const std::vector<std::string> & columns,
            const std::size_t column,
            const std::size_t line_number
        ) {
            if (column >= columns.size()) {
                throw std::runtime_error("Missing data column at line " + std::to_string(line_number));
            }
            std::istringstream stream(columns[column]);
            double value = 0.0;
            if (!(stream >> value)) {
                throw std::runtime_error("Invalid numeric value at line " + std::to_string(line_number));
            }
            std::string extra;
            if (stream >> extra) {
                throw std::runtime_error("Invalid numeric value at line " + std::to_string(line_number));
            }
            return value;
        }

        void Validate() const {
            if (input_dim_ == 0 || output_dim_ == 0) {
                throw std::invalid_argument("Data dimensions must be positive.");
            }
            if (observations_.empty()) {
                throw std::invalid_argument("DataSet must contain at least one observation.");
            }
            for (const Observation & observation : observations_) {
                if (observation.x.size() != input_dim_ || observation.y.size() != output_dim_) {
                    throw std::invalid_argument("Observation shape does not match DataSet dimensions.");
                }
            }
        }

        std::size_t input_dim_ = 0;
        std::size_t output_dim_ = 0;
        std::vector<Observation> observations_;
};

} // namespace bayes_emc
