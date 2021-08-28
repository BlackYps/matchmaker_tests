import asyncio
import base64
import inspect
import itertools
import json
import os
import random
import re
import signal
import string
import struct
import subprocess
import sys
import textwrap
from asyncio import IncompleteReadError, StreamReader, StreamWriter
from collections import defaultdict, deque, namedtuple
from enum import Enum, auto
from functools import partial
from getpass import getpass
from hashlib import sha256
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional,
                    Tuple, Union)

try:
    import readline
except ImportError:
    def nop(*args, **kwargs):
        """Do nothing function."""

    class Dummy(object):
        """Do nothing object."""
        def __getattribute__(self, name):
            return nop

    readline = Dummy()

USER_AGENT = "askaholics-faf-cli"
VERSION = "1.0.0"

OUTPUT_WIDTH = 80

#
# Type Aliases
#

Argument = Union[str, int, Dict[Union[str, int], Union[str, int]], List[Union[str, int]]]

#
# Helpers
#
signal.signal(signal.SIGINT, lambda _1, _2: os._exit(1))

# For REPL commands entered by the user
COMMANDS = {}
ALIASES = {}
# For commands sent by the server
HANDLERS = {}


def cmd(name: str, *aliases) -> Callable:
    def decorator(func: Callable) -> Callable:
        COMMANDS[name] = func
        for alias in aliases:
            assert alias not in COMMANDS, \
                f"Cannot create alias '{alias}' because there is already a command with that name!"
            ALIASES[alias] = name
        return func
    return decorator


def get_command(name: str) -> Optional[Callable]:
    name = _command_from_alias(name)
    return COMMANDS.get(name)


def _command_from_alias(alias: str) -> str:
    if alias in ALIASES:
        return ALIASES[alias]
    return alias


def handler(name: str) -> Callable:
    def decorator(func: Callable):
        HANDLERS[name] = func
        return func
    return decorator


def get_handler(name: str) -> Optional[Callable]:
    return HANDLERS.get(name)


def try_int(value: Any) -> Any:
    try:
        value = int(value)
    except ValueError:
        pass
    return value


async def ainput(prompt: str) -> str:
    """ Async wrapper around builtin function 'input'. """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(input, prompt))

#######################
#     Input Parser    #
#######################

Token = namedtuple("Token", ["data", "category", "offset"])
Command = namedtuple("Command", ["name", "args"])


class TokenCategory(Enum):
    MALFORMED = auto()
    IDENTIFIER = auto()
    PUNCTUATION = auto()
    NUMERIC_LITERAL = auto()
    BOOLEAN_LITERAL = auto()


class ParseError(Exception):
    def __init__(
        self,
        eof: bool,
        token: Optional[Token],
        expected: Optional[str] = None
    ) -> None:
        self.eof = eof
        self.token = token
        self.expected = expected


def parse(data: str) -> Any:
    tokens = tokenize(data)
    return parse_command(tokens)


def parse_command(tokens: Iterator[Token]) -> Command:
    command_str = ""
    args: List[Any] = []
    try:
        command = next(tokens)
        if command.category is not TokenCategory.IDENTIFIER:
            raise ParseError(False, command, expected="command")

        command_str = command.data

        while True:
            args.append(parse_arg(tokens))
    except StopIteration:
        return Command(command_str, args)


def parse_arg(tokens: Iterator[Token]) -> Any:
    token = next(tokens)

    if token.category is TokenCategory.IDENTIFIER:
        return token.data
    elif token.category is TokenCategory.NUMERIC_LITERAL:
        return int(token.data)
    elif token.category is TokenCategory.BOOLEAN_LITERAL:
        return token.data.lower() == "true"
    elif token.category is TokenCategory.PUNCTUATION:
        if token.data == "{":
            return parse_dict(tokens)
        elif token.data == "[":
            return parse_list(tokens)

    raise ParseError(False, token, expected="argument")


def parse_dict(tokens: Iterator[Token]) -> Dict[str, Any]:
    data = {}
    try:
        while True:
            k, v = parse_key_value(tokens)
            if k is None:
                return data
            data[k] = v
    except StopIteration:
        raise ParseError(True, token=None)  # TODO: Signal end of input

    return data


def parse_key_value(tokens: Iterator[Token]) -> Tuple[Optional[Union[str, int]], Any]:
    key = next(tokens)
    if key.data == ",":
        key = next(tokens)
    elif key.data == "}":
        return None, None

    if key.category not in (
        TokenCategory.IDENTIFIER,
        TokenCategory.NUMERIC_LITERAL
    ):
        raise ParseError(False, key, expected="dictionary key")

    colon = next(tokens)
    if not colon.data == ":":
        raise ParseError(False, colon, expected=":")

    value_data = parse_arg(tokens)

    if key.category == TokenCategory.NUMERIC_LITERAL:
        key_data = int(key.data)
    elif key.category is TokenCategory.BOOLEAN_LITERAL:
        key_data = key.data.lower() == "true"
    else:
        key_data = key.data

    return key_data, value_data


def parse_list(tokens: Iterator[Token]) -> List[Any]:
    data = []

    try:
        while True:
            item = parse_list_item(tokens)
            if item is None:
                return data
            data.append(item)
    except StopIteration:
        raise ParseError(True, token=None)

    return data


def parse_list_item(tokens: Iterator[Token]) -> Any:
    item = next(tokens)
    if item.data == ",":
        item = next(tokens)
    elif item.data == "]":
        return None

    if item.category == TokenCategory.NUMERIC_LITERAL:
        return int(item.data)
    elif item.category == TokenCategory.BOOLEAN_LITERAL:
        return item.data.lower() == "true"
    return item.data


def tokenize(data: str) -> Generator[Token, None, None]:
    token: List[str] = []
    data_len = len(data)
    i = 0
    curr_state = "start"
    states = {}

    PUNCTUATION = "[]{},:"
    IDENT_STARTERS = string.ascii_letters + "$"
    BACKSLASH_ESCAPES = {
        "t": "\t",
        "n": "\n",
        "r": "\r"
    }

    def state(func: Callable) -> Callable:
        "Decorator for adding a state handler"
        states[func.__name__] = func
        return func

    def next_char() -> str:
        nonlocal i
        char = data[i]
        i += 1
        return char

    def peek_char() -> Optional[str]:
        if i >= data_len:
            return None
        return data[i]

    def set_state(name: str) -> None:
        nonlocal curr_state
        curr_state = name

    def make_token(category: TokenCategory) -> Token:
        return Token("".join(token), category, offset)

    def finish(category: TokenCategory) -> Token:
        set_state("stop")
        return make_token(category)

    @state
    def start() -> Optional[Token]:
        char = next_char()
        token.append(char)

        if char in IDENT_STARTERS:
            return set_state("ident")
        if char in string.digits+"-+":
            return set_state("numlit")
        if char in "\"'":
            return set_state("strlit")
        if char in PUNCTUATION:
            return finish(TokenCategory.PUNCTUATION)

        return finish(TokenCategory.MALFORMED)

    @state
    def ident() -> Optional[Token]:
        "Identifiers take anything that is not whitespace"
        char = peek_char()
        if not char or char in string.whitespace or char in PUNCTUATION:
            set_state("stop")
            token_str = "".join(token)
            category = TokenCategory.IDENTIFIER
            if token_str in ("True", "False", "true", "false"):
                category = TokenCategory.BOOLEAN_LITERAL
            return Token(token_str, category, offset)

        token.append(char)
        next_char()

    @state
    def numlit() -> Optional[Token]:
        "Numeric literals can only be integers"
        char = peek_char()

        if not char:
            return finish(TokenCategory.NUMERIC_LITERAL)

        if char in string.digits:
            token.append(char)
            next_char()
        elif char in IDENT_STARTERS:
            token.append(char)
            next_char()
            set_state("ident")
        else:
            return finish(TokenCategory.NUMERIC_LITERAL)

    @state
    def strlit() -> Optional[Token]:
        "String literals can be anything between matching single or double quotes"
        char = peek_char()
        if not char:
            return finish(TokenCategory.MALFORMED)

        next_char()
        if len(token) > 1 and char == token[0]:
            set_state("stop")
            return Token("".join(token[1:]), TokenCategory.IDENTIFIER, offset)

        # Handle backslash escapes
        if char == "\\":
            char = peek_char()
            if not char:
                return finish(TokenCategory.MALFORMED)
            next_char()
            char = BACKSLASH_ESCAPES.get(char, char)
        token.append(char)

    while True:
        offset = i
        # Skip whitespace
        char = peek_char()
        if not char:
            break
        if char in string.whitespace:
            next_char()
            continue

        token = []
        curr_state = "start"
        while curr_state != "stop":
            # When the stop state is called, the handler should return a `Token`
            ret = states[curr_state]()
        yield ret


#######################
#   End Input Parser  #
#######################


class FafClient(object):
    PROMPT_DISCONNECTED = ">> "
    PROMPT_CONNECTED = "++ "
    PROMPT_CONTINUE_INPUT = ".. "

    def __init__(self):
        self.stdout = None
        self.stdin = None
        self.proto = None
        self.server_commands_buffer = deque()
        self.prompt = FafClient.PROMPT_DISCONNECTED

        self.working_data = {}
        self.data_categories = defaultdict(set)
        self.message_waiters = defaultdict(asyncio.Future)

        self._completion_iter = None
        self.faf_uid_path = "faf-uid"

    #
    # REPL Core implementation
    #

    async def repl(self) -> None:
        try:
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self.readline_completer)
            readline.set_auto_history(False)

            print("Welcome to FAF cli!")
            print("Type 'help' for a list of available commands.")

            continue_input = False
            command = ""
            while True:
                if continue_input:
                    line = await ainput(FafClient.PROMPT_CONTINUE_INPUT) or ""
                    command += line.strip()
                else:
                    line = await ainput(self.prompt) or ""
                    command = line.strip()

                if not command:
                    continue

                try:
                    cmd = parse(command)
                    readline.add_history(command)
                except ParseError as e:
                    if e.eof:
                        continue_input = True
                        continue

                    assert e.token is not None
                    print("Invalid Syntax")
                    print(f"    {command}")
                    print(f"    {' ' * e.token.offset}^ Expected {e.expected}")
                    continue_input = False
                    readline.add_history(command)
                    continue

                await self._handle_command(cmd)
                continue_input = False
        except EOFError:
            print("Bye!")
        finally:
            await self._cleanup()

    async def _handle_command(self, command: Command):
        try:
            cmd, args = command.name, command.args

            def eval_variables(data: Union[str, list, dict]) -> Union[str, list, dict]:
                if isinstance(data, str) and data.startswith("$"):
                    return self.working_data[data[1:]]
                elif isinstance(data, list):
                    return list(map(eval_variables, data))
                elif isinstance(data, dict):
                    return {
                        k: eval_variables(v)
                        for k, v in data.items()
                    }

                return data

            args = list(map(eval_variables, args))
            handler = get_command(cmd)
            if not handler:
                print(f"{cmd}: command not found")
                return

            await handler(self, *args)
        except TypeError as e:
            msg = str(e)
            match = re.match(r".*\(\) (.*)", msg)
            if match:
                msg = match.group(1)
            print(msg)
            self._print_help_for(cmd)
        except EOFError as e:
            raise e
        except Exception as e:
            print("error:", e)

    async def _cleanup(self):
        if self.stdout:
            self.stdout.close()
        self._disconnect_proto(display=False)

    async def _read_from_proto(self):
        assert self.proto

        try:
            while self.proto:
                msg = await self.proto.read_message()
                self.server_commands_buffer.append(msg)
                await self._process_server_message(msg)
        except IncompleteReadError:
            self._disconnect_proto()
        except Exception as e:
            print("error:", e)
            self._disconnect_proto()

    def _disconnect_proto(self, display=True):
        if self.proto:
            self.proto.close()
        self.proto = None
        self.working_data = {}
        self.data_categories = defaultdict(set)
        self.prompt = FafClient.PROMPT_DISCONNECTED
        if display:
            buffer = readline.get_line_buffer()
            print(f"\r{self.prompt}{buffer}", end='')
            sys.stdout.flush()

    def readline_completer(self, text: str, state: int):
        if state == 0:
            buffer = readline.get_line_buffer()
            self._completion_iter = self._yield_completion_options(buffer)

        if not self._completion_iter:
            return

        try:
            return next(self._completion_iter)
        except StopIteration:
            return

    def _yield_completion_options(self, text: str):
        cmd, *args = text.split()  # TODO: replace with parse args
        if not args and not text.endswith(" "):
            options = self.__completion_options_command_name(cmd)
        else:
            options = self.__completion_options_command_args(text, cmd, args)
        options.sort()
        for option in options:
            yield option

    def __completion_options_command_name(self, text: str):
        return list(
            filter(lambda t: t.startswith(text), itertools.chain(COMMANDS.keys(), ALIASES.keys()))
        )

    def __completion_options_command_args(self, text: str, cmd: str, args: List):
        func = get_command(cmd)
        if not func:
            return []
        spec = inspect.getfullargspec(func)
        spec_args = list(filter(lambda a: a != 'self', spec.args))
        complete_index = len(args) if text.endswith(' ') else len(args) - 1
        if complete_index >= len(spec_args):
            return []
        category = spec_args[complete_index]
        to_complete = "" if not args else args[complete_index]

        if category == "command":
            return self.__completion_options_command_name(to_complete)
        # We can only auto complete certain args
        if category not in self.data_categories:
            return []
        if not args:
            return ["$"]
        options = list(
            filter(
                lambda t: t.startswith(to_complete),
                map(lambda o: "$" + o, self.data_categories[category])
            )
        )
        # Readline does not like the $, so we need to remove it here. This does
        # not happen for other characters
        return list(map(lambda o: o[1:], options))

    async def _process_server_message(self, msg):
        command = msg.get("command")
        if not command:
            return

        if command in self.message_waiters:
            self.message_waiters[command].set_result(msg)

        handler = get_handler(command)
        if not handler:
            return

        # A value error here means the code has a bug
        await handler(self, msg)

    async def wait_server_message(self, command: str, timeout: float = 5):
        """ Wait for a message to arrive, or timeout. Returns the message """
        try:
            fut = self.message_waiters[command]
            await asyncio.wait_for(fut, timeout)
            return fut.result()
        finally:
            # Weird things can happen if multiple coros try to clean up the same
            # future
            if command in self.message_waiters:
                del self.message_waiters[command]

    async def wait_server_message_or_error_message(
        self, command: str, *err_commands: str, timeout: float = 5
    ):
        """
        Wait for a server message to arrive, or for a different message
        signaling that the first message will never arrive. Returns the message
        """
        try:
            done, _ = await asyncio.wait(
                [
                    self.message_waiters[command],
                    *[self.message_waiters[cmd] for cmd in err_commands]
                ],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            if done:
                return done.pop().result()
        finally:
            # Weird things can happen if multiple coros try to clean up the same
            # future
            for cmd in (command,) + err_commands:
                if cmd in self.message_waiters:
                    del self.message_waiters[cmd]

    async def call_faf_uid(self, session: str):
        loop = asyncio.get_event_loop()
        process = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [self.faf_uid_path, str(session)],
                stdout=subprocess.PIPE
            )
        )
        return process.stdout

    # # # # # # # # #
    # REPL Commands #
    # # # # # # # # #

    @cmd('help', 'h')
    async def _help(self, command: Optional[str] = None):
        """ Show the help text for a command. """
        if command is not None and not isinstance(command, str):
            return
        command = _command_from_alias(command)
        alias_lookup = self.__make_alias_lookup_table()
        func = get_command(command)
        if command and func:
            self.__print_help_for(command, func, alias_lookup)
            return

        print(textwrap.fill(
            """ As the server responds with commands, FAF cli will gather useful
            information which can be viewed with the 'data' command. As a
            convenience, this data can be used as variables in subsequent
            commands.""".replace("    ", ""),
            width=OUTPUT_WIDTH
        ))
        print()
        print("Help format: command, aliases {<arg>} {[optional_arg]}")
        for cmd, fn in COMMANDS.items():
            self.__print_help_for(cmd, fn, alias_lookup)

    def _print_help_for(self, command: str):
        """ Prints the help text for specific command"""
        if command not in COMMANDS:
            return
        alias_lookup = self.__make_alias_lookup_table()
        self.__print_help_for(command, COMMANDS[command], alias_lookup)

    def __make_alias_lookup_table(self) -> Dict[str, List[str]]:
        alias_lookup = defaultdict(list)
        for alias, name in ALIASES.items():
            alias_lookup[name].append(alias)
        return alias_lookup

    def __print_help_for(self, command, func: Callable, alias_lookup: Dict):
        spec = inspect.getfullargspec(func)
        num_args = len(spec.args) - len(spec.defaults or [])
        args = []
        args += list(map(lambda a: f"<{a}>", filter(lambda x: x != 'self', spec.args[:num_args])))
        if spec.varargs:
            args.append(f"*{spec.varargs}")
        args += list(map(lambda a: f"[{a}]", spec.args[num_args:]))
        print(" " * 3, ", ".join([command, *alias_lookup[command]]), *args)
        doc = func.__doc__
        if not doc:
            return
        doc = textwrap.wrap(doc.strip(), width=OUTPUT_WIDTH-7)
        for doc_line in doc:
            print(" " * 7, doc_line)

    @cmd('exit')
    async def _exit(self):
        """ Take a guess... """
        raise EOFError()

    @cmd('data')
    async def _print_working_data(self):
        """ View your current working data. These values can be used as parameters
        to other commands. """
        for k, v in self.working_data.items():
            print(f"    ${k} = {v}")

    @cmd('set')
    async def _set_working_data_var(self, key: str, value: Argument):
        """ Write a value to your current working data. """
        self.working_data[key] = value

    @cmd('echo', 'print')
    async def _echo(self, *args: Argument):
        """ Proxy for builtin `print`. """
        print(*args)

    @cmd('connect', 'c')
    async def _connect(self, host: str, port: str = 8001):
        """ Establish a connection to the faf server. """
        if self.proto:
            self._disconnect_proto()
        print("WARNING: Connection is not encrypted!")
        reader, writer = await asyncio.open_connection(host, int(port))
        self.proto = QDataStreamProtocol(reader, writer)
        self.prompt = FafClient.PROMPT_CONNECTED
        asyncio.ensure_future(self._read_from_proto())

    @cmd('disconnect', 'dc')
    async def _disconnect(self):
        """ Close the server connection. """
        self._disconnect_proto(display=False)

    @cmd('messages')
    async def _print_server_messages(self, command_filter: str = None):
        """ View messages sent by the server. """
        while self.server_commands_buffer:
            message = self.server_commands_buffer.popleft()
            if command_filter:
                command = message.get('command')
                if command_filter != command:
                    continue

            print(message)

    @cmd('message')
    async def _print_server_message(self):
        """ View the next message sent by the server. """
        if not self.server_commands_buffer:
            return

        print(self.server_commands_buffer.popleft())

    @cmd('command', 'cmd')
    async def _send_server_command(self, message: Union[dict, str]):
        """ Send a command to the server. 'message' can be a dictionary or a string. """
        if not self.proto:
            raise Exception("Not connected to a server")

        try:
            self.proto.send_message(message)
            await self.proto.drain()
        except IncompleteReadError:
            print("Connection aborted")

    @cmd('test')
    async def _connect_to_test(self):
        """ Connect to the test server. """
        await self._connect("lobby.test.faforever.com", 8001)

    @cmd('ask_session', 'session')
    async def _command_ask_session(self):
        """ Send 'ask_session' command. """
        await self._send_server_command({
            "command": "ask_session",
            "user_agent": USER_AGENT,
            "version": VERSION
        })

    @cmd('hello')
    async def _command_hello(self, login: str, password: str, unique_id: str=None):
        """ Send 'hello' command. WARNING: Your password will appear on the screen! """
        await self._send_server_command({
            "command": "hello",
            "login": login,
            "password": sha256(password.encode()).hexdigest(),
            "unique_id": unique_id
        })

    @cmd('game_host', 'host')
    async def _command_game_host(self, title: str = None):
        """ Send 'game_host' command. """
        await self._send_server_command({
            "command": "game_host",
            "visibility": "public",
            "title": title
        })

    @cmd('game_join', 'join')
    async def _command_game_join(self, game_id: str):
        """ Send 'game_join' command. """
        await self._send_server_command({
            "command": "game_join",
            "uid": try_int(game_id)
        })

    @cmd('game_matchmaking')
    async def _command_game_matchmaking(
        self, state: str, faction: str = 'uef', mod: str = 'ladder1v1'
    ):
        """ Send 'game_matchmaking' command. """
        await self._send_server_command({
            "command": "game_matchmaking",
            "state": state,
            "mod": mod,
            "faction": try_int(faction)
        })

    @cmd('avatar')
    async def _command_avatar(self, action: str, url: str = None):
        """ Send 'avatar' command. """
        await self._send_server_command({
            "command": "avatar",
            "action": action,
            "avatar": url
        })

    @cmd('avatar_list')
    async def _command_avatar_list(self):
        """ Shorthand for sending `avatar` command with `list_avatar` action. """
        await self._command_avatar("list_avatar")

    @cmd('invite_to_party')
    async def _command_invite_to_party(self, player_id: str):
        await self._send_server_command({
            "command": "invite_to_party",
            "recipient_id": int(player_id)
        })

    @cmd('accept_party_invite')
    async def _command_accept_party_invite(self, player_id: str):
        await self._send_server_command({
            "command": "accept_party_invite",
            "sender_id": int(player_id)
        })

    @cmd('kick_player_from_party')
    async def _command_kick_player_from_party(self, player_id: str):
        await self._send_server_command({
            "command": "kick_player_from_party",
            "kicked_player_id": int(player_id)
        })

    @cmd('leave_party')
    async def _command_leave_party(self):
        await self._send_server_command({"command": "leave_party"})

    @cmd('ready_party')
    async def _command_ready_party(self):
        await self._send_server_command({"command": "ready_party"})

    @cmd('unready_party')
    async def _command_unready_party(self):
        await self._send_server_command({"command": "unready_party"})

    #
    # GPGNet Commands
    #

    @cmd('gpg_command', 'gpg')
    async def _send_gpgnet_command(self, command: str, *args: str):
        """ Send a GPGNet command. """
        await self._send_server_command({
            "target": "game",
            "command": command,
            "args": args
        })

    @cmd('gpg_gamestate')
    async def _gpg_gamestate(self, state: str):
        """ Send a GPGNet GameState command. States should be one of
            {Idle, Lobby, Launching, Ended} """
        await self._send_gpgnet_command("GameState", state)

    @cmd('gpg_gameoption')
    async def _gpg_gameoption(self, key: str, value: str):
        """ Send a GPGNet GameOption command. """
        value = try_int(value)
        await self._send_gpgnet_command("GameOption", key, value)

    @cmd('gpg_playeroption')
    async def _gpg_playeroption(self, player_id: str, key: str, value: Any):
        """ Send a GPGNet PlayerOption command. """
        value = try_int(value)
        await self._send_gpgnet_command("PlayerOption", int(player_id), key, value)

    @cmd('gpg_gameresult')
    async def _gpg_gameresult(self, army: str, result: str, score: str):
        """ Send a GPGNet GameResult command. 'result' should be one of
        {score, defeat, victory, draw} """
        army = try_int(army)
        await self._send_gpgnet_command("GameResult", army, f"{result} {score}")

    @cmd('gpg_jsonstats')
    async def _gpg_jsonstats(self, stats: dict):
        """ Send a GPGNet JsonStats command. 'stats' is a dictionary. """
        await self._send_gpgnet_command("JsonStats", json.dumps(stats))
    #
    # Sequence Commands. REPL commands that send more than one protocol command
    #

    @cmd('login')
    async def _seq_login(self, username: str, password: str = None):
        """ Perform the login sequence with the server. """
        if not password:
            password = getpass()
        await self._command_ask_session()
        await self.wait_server_message_or_error_message("session", "update", 10)
        session = self.working_data.get("session")
        if not session:
            session = int(random.randrange(0, 4294967295))
            print(
                "warning: unable to obtain a session id, using",
                session, "instead"
            )
        unique_id = await self.call_faf_uid(str(session))
        await self._command_hello(username, password, unique_id.decode())

    @cmd('seq_launch_fa')
    async def _seq_launch_fa(self):
        """ Send the GPGNet messages to simulate fa opening. """
        await self._gpg_gamestate("Idle")
        await self._gpg_gamestate("Lobby")

    @cmd('seq_game_host')
    async def _seq_game_host(self, player_id: str = None):
        """ Host a game and simulate fa opening. """
        player_id = player_id or self.working_data.get('id')
        if player_id is None:
            raise Exception("$id not set, you must provide a 'player_id'")
        await self._command_game_host()
        await self._seq_launch_fa()
        await self._seq_configure_joining_player(player_id, 1)

    @cmd('seq_game_join')
    async def _seq_game_join(self, game_id: str):
        """ Join a game and simulate fa opening. Note that the host will need to
        configure the PlayerOptions """
        await self._command_game_join(game_id)
        await self._seq_launch_fa()

    @cmd('seq_configure_joining_player')
    async def _seq_configure_joining_player(self, player_id: str, army: str):
        """ Configure PlayerOption needed for the game to launch successfully.
        This should only work when the host is sending the commands. """
        army = int(army)
        await self._gpg_playeroption(player_id, "Team", 1)
        await self._gpg_playeroption(player_id, "Army", army)
        await self._gpg_playeroption(player_id, "Faction", 1)
        await self._gpg_playeroption(player_id, "Color", army)
        await self._gpg_playeroption(player_id, "StartSpot", army)

    @cmd('seq_end_game_1v1')
    async def _seq_end_game_1v1(self, player1: str, player2: str):
        """ Report some game stats and send GameState == Ended. You should run
        `gpg_gameresult` first to prevent the game from being marked as invalid. """
        await self._gpg_jsonstats(json_stats_1v1(player1, player2))
        await self._gpg_gamestate("Ended")

    #
    # Server message handlers
    #

    async def __set_login(self, message):
        player_id = message.get("id")
        player_name = message.get("login")
        if player_name and player_name:
            self.working_data[player_name] = player_id
            self.data_categories["player_id"].add(player_name)

    @handler('session')
    async def _handle_session(self, message):
        session = message.get("session")
        if session:
            self.working_data["session"] = session

    @handler('welcome')
    async def _handle_welcome(self, message):
        me = message.get("me")
        if not me:
            return
        player_id = me.get("id")
        if player_id:
            self.working_data["id"] = player_id
            self.data_categories["player_id"].add("id")
        await self.__set_login(me)

    @handler('player_info')
    async def _handle_player_info(self, message):
        players = message.get("players")
        if not players:
            return
        for player in players:
            await self.__set_login(player)

#######################
# End class FafClient #
#######################


class MultiFafClient(object):
    """Run multiple faf clients at the same time."""

    def __init__(self, amount: int):
        self.clients = [FafClient() for _ in range(amount)]

    async def repl(self) -> None:
        try:
            # TODO: Add tab completion for multi client?
            # readline.parse_and_bind("tab: complete")
            # readline.set_completer(self.readline_completer)
            readline.set_auto_history(False)

            print("Welcome to FAF cli multi-client mode!")
            print("Running", len(self.clients), "clients simultaneously.")

            continue_input = False
            command = ""
            while True:
                if continue_input:
                    line = await ainput(FafClient.PROMPT_CONTINUE_INPUT) or ""
                    command += line.strip()
                else:
                    line = await ainput(FafClient.PROMPT_DISCONNECTED) or ""
                    command = line.strip()

                if not command:
                    continue

                try:
                    cmd = parse(command)
                    readline.add_history(command)
                except ParseError as e:
                    if e.eof:
                        continue_input = True
                        continue

                    assert e.token is not None
                    print("Invalid Syntax")
                    print(f"    {command}")
                    print(f"    {' ' * e.token.offset}^ Expected {e.expected}")
                    continue_input = False
                    readline.add_history(command)
                    continue

                # Hack for getting different login names
                if cmd.name == "login":
                    client = next(
                        (c for c in self.clients
                            if "id" not in c.working_data and c.proto),
                        None
                    )
                    if client:
                        await client._handle_command(cmd)
                elif cmd.name in ("echo", "help") and self.clients:
                    await self.clients[0]._handle_command(cmd)
                else:
                    await asyncio.gather(
                        *[client._handle_command(cmd) for client in self.clients]
                    )
                continue_input = False
        except EOFError:
            print("Bye!")
        finally:
            await self._cleanup()

    async def _cleanup(self):
        await asyncio.gather(
            *[client._cleanup() for client in self.clients]
        )


# Sample JsonStats


def json_stats_1v1(login_1: str, login_2: str):
    return {
      "stats": [
        {
          "faction": 1,
          "type": "Human",
          "name": login_1,
          "general": {
            "score": 5370,
            "currentcap": {
              "count": 1000
            },
            "kills": {
              "mass": 18483.279296875,
              "count": 11,
              "energy": 5002463
            },
            "built": {
              "mass": 532,
              "count": 19,
              "energy": 4160
            },
            "lost": {
              "mass": 18000,
              "count": 1,
              "energy": 5000000
            },
            "energy": 10129.700195313,
            "currentunits": {
              "count": 1.8000001907349
            },
            "mass": 1004.1658935547
          },
          "units": {
            "air": {"built": 0, "kills": 0, "lost": 0},
            "land": {"built": 0, "kills": 0, "lost": 0},
            "naval": {"built": 0, "kills": 0, "lost": 0},
            "experimental": {"built": 0, "kills": 0, "lost": 0},
            "tech1": {"built": 0, "kills": 0, "lost": 0},
            "tech2": {"built": 0, "kills": 0, "lost": 0},
            "tech3": {"built": 0, "kills": 0, "lost": 0},
            "engineer": {"built": 0, "kills": 0, "lost": 0},
            "transportation": {"built": 0, "kills": 0, "lost": 0},
            "sacu": {"built": 0, "kills": 0, "lost": 0},
            "cdr": {"built": 1, "kills": 1, "lost": 0}
          },
          "blueprints": {
            "ual0001": {"built": 1, "kills": 1, "lost": 0}
          }
        },
        {
          "faction": 1,
          "type": "Human",
          "name": login_2,
          "general": {
            "score": 10,
            "currentcap": {
              "count": 1000
            },
            "kills": {
              "mass": 1,
              "count": 11,
              "energy": 500
            },
            "built": {
              "mass": 5,
              "count": 1,
              "energy": 41
            },
            "lost": {
              "mass": 180,
              "count": 1,
              "energy": 50000
            },
            "energy": 101,
            "currentunits": {
              "count": 1
            },
            "mass": 100
          },
          "units": {
            "air": {"built": 0, "kills": 0, "lost": 0},
            "land": {"built": 0, "kills": 0, "lost": 0},
            "naval": {"built": 0, "kills": 0, "lost": 0},
            "tech1": {"built": 0, "kills": 0, "lost": 0},
            "tech2": {"built": 0, "kills": 0, "lost": 0},
            "tech3": {"built": 0, "kills": 0, "lost": 0},
            "experimental": {"built": 0, "kills": 0, "lost": 0},
            "engineer": {"built": 0, "kills": 0, "lost": 0},
            "transportation": {"built": 0, "kills": 0, "lost": 0},
            "sacu": {"built": 0, "kills": 0, "lost": 0},
            "cdr": {"built": 1, "kills": 0, "lost": 1}
          },
          "blueprints": {
            "ual0001": {"built": 1, "kills": 0, "lost": 1}
          }
        }
      ]
    }


# End Sample JsonStats


class QDataStreamProtocol(object):
    """
    Implements the legacy QDataStream-based encoding scheme
    """
    def __init__(self, reader: StreamReader, writer: StreamWriter):
        """
        Initialize the protocol

        :param StreamReader reader: asyncio stream to read from
        """
        self.reader = reader
        self.writer = writer

    @staticmethod
    def read_qstring(buffer: bytes, pos: int=0) -> Tuple[int, str]:
        """
        Parse a serialized QString from buffer (A bytes like object) at given position

        Requires len(buffer[pos:]) >= 4.

        Pos is added to buffer_pos.

        :type buffer: bytes
        :return (int, str): (buffer_pos, message)
        """
        chunk = buffer[pos:pos + 4]
        rest = buffer[pos + 4:]
        assert len(chunk) == 4

        (size, ) = struct.unpack('!I', chunk)
        if len(rest) < size:
            raise ValueError(
                "Malformed QString: Claims length {} but actually {}. Entire buffer: {}"
                .format(size, len(rest), base64.b64encode(buffer)))
        return size + pos + 4, (buffer[pos + 4:pos + 4 + size]).decode('UTF-16BE')

    @staticmethod
    def read_int32(buffer: bytes, pos: int=0) -> Tuple[int, int]:
        """
        Read a serialized 32-bit integer from the given buffer at given position

        :return (int, int): (buffer_pos, int)
        """
        chunk = buffer[pos:pos + 4]
        assert len(chunk) == 4

        (num, ) = struct.unpack('!i', chunk)
        return pos + 4, num

    @staticmethod
    def pack_qstring(message: str) -> bytes:
        encoded = message.encode('UTF-16BE')
        return struct.pack('!i', len(encoded)) + encoded

    @staticmethod
    def pack_block(block: bytes) -> bytes:
        return struct.pack('!I', len(block)) + block

    @staticmethod
    def read_block(data):
        buffer_pos = 0
        while len(data[buffer_pos:]) > 4:
            buffer_pos, msg = QDataStreamProtocol.read_qstring(data, buffer_pos)
            yield msg

    @staticmethod
    def pack_message(*args: str) -> bytes:
        """
        For sending a bunch of QStrings packed together in a 'block'
        """
        msg = bytearray()
        for arg in args:
            if not isinstance(arg, str):
                raise NotImplementedError("Only string serialization is supported")

            msg += QDataStreamProtocol.pack_qstring(arg)
        return QDataStreamProtocol.pack_block(msg)

    async def read_message(self):
        """
        Read a message from the stream

        On malformed stream, raises IncompleteReadError

        :return dict: Parsed message
        """
        (block_length, ) = struct.unpack('!I', (await self.reader.readexactly(4)))
        block = await self.reader.readexactly(block_length)
        # FIXME: New protocol will remove the need for this

        pos, action = self.read_qstring(block)
        if action in ['UPLOAD_MAP', 'UPLOAD_MOD']:
            pos, _ = self.read_qstring(block, pos)  # login
            pos, _ = self.read_qstring(block, pos)  # session
            pos, name = self.read_qstring(block, pos)
            pos, info = self.read_qstring(block, pos)
            pos, size = self.read_int32(block, pos)
            data = block[pos:size]
            return {
                'command': action.lower(),
                'name': name,
                'info': json.loads(info),
                'data': data
            }
        elif action in ['PING', 'PONG']:
            return {
                'command': action.lower()
            }
        else:
            message = json.loads(action)
            try:
                for part in self.read_block(block):
                    try:
                        message_part = json.loads(part)
                        if part != action:
                            message.update(message_part)
                    except (ValueError, TypeError):
                        if 'legacy' not in message:
                            message['legacy'] = []
                        message['legacy'].append(part)
            except (KeyError, ValueError):
                pass
            return message

    async def drain(self):
        await self.writer.drain()

    def close(self):
        self.writer.close()

    def send_message(self, message: dict):
        self.writer.write(self.pack_message(json.dumps(message)))

    def send_messages(self, messages):
        payload = [self.pack_message(json.dumps(msg)) for msg in messages]
        self.writer.writelines(payload)

    def send_raw(self, data):
        self.writer.write(data)

#################################
# End Class QDataStreamProtocol #
#################################


def main(amount=1):
    loop = asyncio.get_event_loop()

    if amount == 1:
        client = FafClient()
    else:
        client = MultiFafClient(amount=amount)

    loop.run_until_complete(client.repl())


if __name__ == '__main__':
    amount = 1
    if len(sys.argv) > 1:
        amount = int(sys.argv[1])
        if amount < 1:
            print("Number of clients must be positive")
            sys.exit()

    main(amount)
