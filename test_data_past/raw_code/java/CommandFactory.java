package filesprocessing;

import java.util.*;

/**
 * The CommandFactory class handles the creation of a Command array from a given input.
 *
 * @author jherskow
 * @author felber
 */
class CommandFactory {
	private static final String
			BAD_COMMAND_FILE = "Bad command file.",
			FILTER ="FILTER", ORDER ="ORDER", DEFAULT_ORDER ="abs";

	/**
	 * This method parses the filter file into an array of Command objects, each with an order and filter.
	 * @param linesArray An array of Strings, each representing a line in the original text file.
	 * @return A new array, containing all commands, in order.
	 * @throws Type2Error If is an insoluble problem with the input.
	 */
	static Command[] makeCommandsFromLines(String[] linesArray)throws Type2Error{
		Command[] commands = new Command[linesArray.length];
		int i=0, commandCounter = 0;
		int filterNum, orderNum;
		String filterArg, orderArg;

		// Iterate over command sets in the filter file.
		while (i<linesArray.length) {
			// Ensure first line is filter
			if (!linesArray[i].equals(FILTER)) throw new Type2Error(BAD_COMMAND_FILE);
			// Create filter from the next line
			filterArg = linesArray[i + 1];
			filterNum =i+1;
			// Ensure the line after that exists, and is ORDER
			if (i+2 >= linesArray.length || !linesArray[i + 2].equals(ORDER)){
				throw new Type2Error(BAD_COMMAND_FILE);
			}
			// If the next line (supposed to be the order parameter) does not exist,
			// Or if it is FILTER - use default order, increment i, and start again from FILTER
			if (i+3 >= linesArray.length || linesArray[i + 3].equals(FILTER)){
				orderArg = DEFAULT_ORDER;
				orderNum =i+2;
				commands[commandCounter] = new Command(filterArg, filterNum+1, orderArg, orderNum+1);
				i += 3;
				commandCounter++;
				continue;
			}
			// Otherwise, create the given order, increment i, and start again from FILTER.
			orderArg = linesArray[i + 3];
			orderNum =i+3;
			commands[commandCounter] = new Command(filterArg, filterNum+1, orderArg, orderNum+1);
			i += 4;
			commandCounter++;
		}
		return Arrays.copyOf(commands, commandCounter);
	}
}
